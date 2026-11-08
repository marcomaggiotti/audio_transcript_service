"""Fine-tuning Whisper on the benchmark set (the Training tab of the podcast app): the settings of each part of Hugging
Face's recipe, the runs, and the separate Python that runs them.

The API never imports PyTorch. Training runs in TRAINING_PYTHON (default: this Python), which needs torch and
transformers: the Docker image installs them in /opt/train with --build-arg TRAINING=true. They live in a Python of
their own because transformers 4.57 (the version the ONNX export is verified with) needs an older huggingface-hub than
faster-whisper. One run at a time, as a child process (app/training_worker.py) that writes its progress to the run's
status.json; its data are the clips of the benchmark set (audio + correct text), split into training and test clips.
A finished run is exported to ONNX into TRAINING_DIR/models and joins the models a job can choose
(finetuned/<name>), like an export in MODELS_DIR. Runs are never retried: one cut off by a restart is marked failed.
"""

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import utcnow

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_whisper_onnx.py"
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
INSPECT_TIMEOUT = 300  # seconds: the first call imports PyTorch and transformers
INTERRUPTED = "Interrupted by a restart; start it again"
PROBE = ("import json, os, torch, transformers\n"
         "print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,"
         " 'cuda': torch.cuda.is_available(),"
         " 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',"
         " 'cpus': os.cpu_count()}))")


class TrainingError(ValueError):
    """A request the training can't take (the message says why)."""


class TrainingUnavailable(RuntimeError):
    """No training Python with PyTorch and transformers."""


# ---------------------------------------------------------------- the settings of each part

class TokenizerSettings(BaseModel):
    """WhisperTokenizer: a clip's correct text -> the token ids the model learns to write (the labels). The service's
    models are English-only: no language or task token, the labels start <|startoftranscript|><|notimestamps|>."""

    predict_timestamps: bool = Field(False, description="Labels without <|notimestamps|> (they have no timestamps "
                                                        "either: Whisper would learn to leave them out)")
    max_label_tokens: int = Field(448, ge=8, le=448, description="Clips whose labels are longer are left out "
                                                                  "(448: the decoder's limit)")
    text: Literal["as_written", "normalized"] = Field(
        "as_written", description="as_written: case and punctuation as in the correct text (what Whisper writes); "
                                  "normalized: lower case without punctuation (as the metric compares)")


class ProcessorSettings(BaseModel):
    """WhisperFeatureExtractor, with the tokenizer a WhisperProcessor: 16 kHz audio -> 80 x 3000 log-mel (30 s)."""

    max_audio_seconds: float = Field(30, ge=1, le=30, description="Longer clips are left out (Whisper hears 30 s)")
    do_normalize: bool = Field(False, description="Zero mean and unit variance on the waveform first (Whisper was "
                                                  "trained without)")


class CollatorSettings(BaseModel):
    """DataCollatorSpeechSeq2SeqWithPadding: labels padded with -100 (ignored by the loss)."""

    strip_start_token: bool = Field(True, description="Cut <|startoftranscript|> when every label starts with it: "
                                                      "the model puts it in front of the labels itself")
    pad_to_multiple_of: int | None = Field(None, ge=1, le=64, description="Pad the labels to a multiple of this")


class MetricSettings(BaseModel):
    """compute_metrics: scored on the test clips with generate(), normalized as the benchmark does."""

    name: Literal["wer", "cer"] = Field("wer", description="Word or character error rate; also the metric_for_best_model")


class DataSettings(BaseModel):
    clip_ids: list[str] | None = Field(None, max_length=10_000, description="Clips of the benchmark set; None = all")
    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Share of the clips held out for testing")
    seed: int = Field(42, ge=0, le=2**31 - 1, description="The shuffle before the split")


class TrainingArgs(BaseModel):
    """Seq2SeqTrainingArguments (the names are transformers'); evaluation and saving every eval_steps, generate() for
    the predictions. The defaults suit whisper-tiny.en on a CPU and a small set of clips."""

    learning_rate: float = Field(1e-5, gt=0, le=1e-2)
    per_device_train_batch_size: int = Field(4, ge=1, le=64)
    gradient_accumulation_steps: int = Field(2, ge=1, le=64)
    warmup_steps: int = Field(20, ge=0, le=10_000)
    max_steps: int = Field(200, ge=1, le=20_000)
    eval_steps: int = Field(50, ge=1, le=20_000, description="Also save_steps (they must match for the best model)")
    logging_steps: int = Field(5, ge=1, le=1_000)
    per_device_eval_batch_size: int = Field(4, ge=1, le=64)
    generation_max_length: int = Field(225, ge=8, le=448)
    lr_scheduler_type: Literal["linear", "cosine", "constant_with_warmup"] = "linear"
    weight_decay: float = Field(0.0, ge=0, le=0.5)
    gradient_checkpointing: bool = Field(False, description="Less memory, about 30% slower")
    fp16: bool = Field(False, description="Half precision: on a GPU only")
    load_best_model_at_end: bool = True
    freeze_encoder: bool = Field(True, description="Train only the decoder (the part that writes the words): less "
                                                   "memory and time, less overfitting on a small set")
    seed: int = Field(42, ge=0, le=2**31 - 1)


class TrainingRequest(BaseModel):
    name: str | None = Field(None, max_length=48, description="Part of the fine-tuned model's name")
    base_model: str | None = Field(None, description="A model of GET /config (a fine-tuned one too); None = the "
                                                     "service's own")
    data: DataSettings = Field(default_factory=DataSettings)
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    processor: ProcessorSettings = Field(default_factory=ProcessorSettings)
    collator: CollatorSettings = Field(default_factory=CollatorSettings)
    metric: MetricSettings = Field(default_factory=MetricSettings)
    args: TrainingArgs = Field(default_factory=TrainingArgs)
    export: bool = Field(True, description="Export to ONNX and add it to the models a job can choose")


# ---------------------------------------------------------------- a run

STATUS_KEYS = ("phase", "message", "step", "max_steps", "epoch", "history", "data", "baseline", "final", "best_step",
               "stopped_at", "device", "threads", "versions", "train_seconds", "train_loss", "export_seconds", "encoder")


class TrainingRun(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    status: Literal["queued", "running", "completed", "failed", "stopped"] = "queued"
    phase: str | None = Field(None, description="loading, preparing, baseline, training, final, saving, exporting, done")
    message: str | None = None
    error: str | None = None
    request: TrainingRequest
    base_model: str
    model_id: str = Field(description="The fine-tuned model's id once exported (finetuned/<name>)")
    clips: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stop_requested: bool = False
    exported: bool = Field(False, description="The model is among the ones a job can choose")
    # From the worker's status.json:
    step: int = 0
    max_steps: int = 0
    epoch: float | None = None
    history: list[dict] = Field(default_factory=list, description="Logged every logging_steps (loss, learning_rate) "
                                                                  "and eval_steps (eval_wer, eval_cer, eval_loss)")
    data: dict | None = Field(None, description="Clips for training and testing, and the ones left out (why)")
    baseline: dict | None = Field(None, description="The base model on the test clips")
    final: dict | None = Field(None, description="The trained model (its best checkpoint) on the test clips")
    best_step: int | None = None
    stopped_at: int | None = None
    device: str | None = None
    threads: int | None = None
    versions: dict | None = None
    train_seconds: float | None = None
    train_loss: float | None = None
    export_seconds: float | None = None
    encoder: str | None = Field(None, description="Whether the exported model shares its base's encoder (frozen)")


def slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", text.lower())).strip("-.")[:40]


class TrainingStore:
    """TRAINING_DIR/runs/<id>/: run.json (the API's), request.json and status.json (the worker's), worker.log;
    TRAINING_DIR/models/<name>/: the fine-tuned exports."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.models_dir = self.root / "models"
        self._lock = threading.Lock()

    def setup(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not ID_PATTERN.fullmatch(run_id):
            raise KeyError(run_id)
        return self.runs_dir / run_id

    def save(self, run: TrainingRun) -> None:
        folder = self.run_dir(run.id)
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / "run.json.tmp"
            tmp.write_text(run.model_dump_json(indent=1))
            tmp.replace(folder / "run.json")

    def get(self, run_id: str) -> TrainingRun:
        path = self.run_dir(run_id) / "run.json"
        if not path.exists():
            raise KeyError(run_id)
        run = TrainingRun.model_validate_json(path.read_text())
        status = self.status(run_id)
        return run.model_copy(update={k: status[k] for k in STATUS_KEYS if status.get(k) is not None})

    def status(self, run_id: str) -> dict:
        try:
            return json.loads((self.run_dir(run_id) / "status.json").read_text())
        except (OSError, ValueError):
            return {}

    def list(self) -> list[TrainingRun]:
        runs = []
        for folder in self.runs_dir.glob("*"):
            if ID_PATTERN.fullmatch(folder.name) and (folder / "run.json").exists():
                try:
                    runs.append(self.get(folder.name))
                except (OSError, ValueError, KeyError):
                    logger.warning("Ignoring unreadable training run %s", folder)
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)


# ---------------------------------------------------------------- the manager

class TrainingManager:
    """Queues runs and runs them one at a time in the training Python; answers the parts' previews."""

    def __init__(self, store: TrainingStore, clips, models, python: str, threads: int = 0, work_dir: str | None = None,
                 export_script: Path = EXPORT_SCRIPT, hf_home: str | None = None):
        self.store = store
        self.clips = clips  # app.benchmark.BenchmarkStore: the data
        self.models = models  # app.models.ModelSet: the bases, and where a fine-tuned model goes
        self.python = python
        self.threads = threads
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "whisper-training"
        self.export_script = export_script
        self.hf_home = hf_home
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self.current: str | None = None
        self._availability: dict | None = None
        self._lock = threading.Lock()
        self._stopping = False

    # lifecycle

    def start(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._work, name="training", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True  # a run killed now was interrupted, not failed
        if self._process and self._process.poll() is None:
            self._process.terminate()
        if self._thread:
            self._queue.put("")
            self._thread.join(timeout=5)

    def recover(self) -> None:
        """Runs a restart cut off can't resume: mark them failed."""
        for run in self.store.list():
            if run.status in ("queued", "running"):
                run.status, run.error, run.finished_at = "failed", INTERRUPTED, utcnow()
                self.store.save(run)

    # the training Python

    def env(self) -> dict:
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
               "PATH": f"{Path(self.python).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        if self.hf_home and "HF_HOME" not in os.environ:
            env["HF_HOME"] = self.hf_home  # the base weights Hugging Face downloads, kept between runs
        return env

    def availability(self, refresh: bool = False) -> dict:
        """Whether the training Python has PyTorch and transformers (checked once, in a child process)."""
        with self._lock:
            if self._availability is not None and not refresh:
                return self._availability
        if not shutil.which(self.python) and not Path(self.python).exists():
            result = {"available": False, "python": self.python,
                      "error": f"{self.python} not found: build the image with --build-arg TRAINING=true"}
        else:
            try:
                proc = subprocess.run([self.python, "-c", PROBE], capture_output=True, text=True, timeout=180,
                                      env=self.env(), cwd=ROOT, check=False)
                if proc.returncode == 0:
                    result = {"available": True, "python": self.python, "error": None,
                              **json.loads(proc.stdout.strip().splitlines()[-1])}
                else:
                    last = (proc.stderr.strip().splitlines() or ["no output"])[-1]
                    result = {"available": False, "python": self.python,
                              "error": f"PyTorch and transformers aren't installed in {self.python} ({last}): build "
                                       "the image with --build-arg TRAINING=true"}
            except (OSError, subprocess.TimeoutExpired, ValueError) as e:
                result = {"available": False, "python": self.python, "error": f"{self.python} didn't start: {e}"}
        with self._lock:
            self._availability = result
        return result

    def require(self) -> dict:
        info = self.availability()
        if not info["available"]:
            raise TrainingUnavailable(info["error"])
        return info

    # inputs

    def base(self, model_id: str | None) -> dict:
        info = self.models.find(model_id)
        if info is None:
            have = ", ".join(m.id for m in self.models.list()) or "none"
            raise TrainingError(f"Unknown model '{model_id}' (this service has: {have})")
        return {"id": info.id, "dir": str(info.path)}

    def clip_inputs(self, clip_ids: list[str] | None) -> list[dict]:
        from app.benchmark import NotFound

        try:
            clips = self.clips.eval_clips(clip_ids)
        except NotFound as e:
            raise TrainingError(f"No such clip in the benchmark set: {e.args[0]}")
        return [{"id": c.id, "name": c.name, "audio": str(c.path), "reference": c.reference} for c in clips]

    def inspect(self, kind: str, payload: dict) -> dict:
        """One look at a part (tokenizer, processor, collator), computed by the training Python."""
        self.require()
        try:
            proc = subprocess.run([self.python, "-m", "app.training_worker", "inspect"], input=json.dumps(
                {"kind": kind, **payload}), capture_output=True, text=True, timeout=INSPECT_TIMEOUT, env=self.env(),
                cwd=ROOT, check=False)
        except subprocess.TimeoutExpired:
            raise TrainingError(f"The {kind} didn't answer within {INSPECT_TIMEOUT} s")
        lines = proc.stdout.strip().splitlines()
        try:
            out = json.loads(lines[-1]) if lines else {}
        except ValueError:
            out = {}
        if proc.returncode != 0 or "error" in out or not out:
            detail = out.get("error") or (proc.stderr.strip().splitlines() or ["no output"])[-1]
            raise TrainingError(f"The {kind} failed: {detail}")
        return out

    # runs

    def submit(self, request: TrainingRequest) -> TrainingRun:
        info = self.require()
        base = self.base(request.base_model)
        clips = self.clip_inputs(request.data.clip_ids)
        if len(clips) < 2:
            raise TrainingError("At least 2 clips are needed (one to train on, one to test with); about 50 or more "
                                "make a difference: add lines with Benchmark in a recording")
        if request.args.fp16 and not info.get("cuda"):
            raise TrainingError("fp16 needs a GPU: this service trains on the CPU")
        run_id = uuid.uuid4().hex
        name = slug(request.name or "") or f"{slug(base['id'].rsplit('/', 1)[-1])}-ft"
        name = f"{name}-{run_id[:6]}"
        run = TrainingRun(id=run_id, name=name, request=request, base_model=base["id"], model_id=f"finetuned/{name}",
                          clips=len(clips), max_steps=request.args.max_steps, message="Waiting for its turn")
        self.store.save(run)
        self._queue.put(run.id)
        return run

    def get(self, run_id: str) -> TrainingRun:
        return self.store.get(run_id)

    def list(self) -> list[TrainingRun]:
        return self.store.list()

    def request_stop(self, run_id: str) -> TrainingRun:
        """A running run stops at its next step (then scores, saves and exports what it learned); a queued one never
        starts."""
        run = self.store.get(run_id)
        if run.status == "queued":
            run.status, run.message, run.finished_at = "stopped", "Stopped before it started", utcnow()
        elif run.status == "running":
            (self.store.run_dir(run_id) / "stop").write_text("")
            run.message = "Stopping after the current step"
        run.stop_requested = True
        self.store.save(run)
        return self.store.get(run_id)

    def delete(self, run_id: str) -> None:
        """The run, its files and its fine-tuned model; a running one is killed first."""
        run = self.store.get(run_id)
        if run.status == "running" and self.current == run_id and self._process and self._process.poll() is None:
            run.status, run.error = "failed", "Deleted while running"
            self.store.save(run)
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if run.exported:
            self.models.remove(run.model_id)
        shutil.rmtree(self.store.models_dir / run.name, ignore_errors=True)
        shutil.rmtree(self.work_dir / run_id, ignore_errors=True)
        self.store.delete(run_id)

    def _work(self) -> None:
        while (run_id := self._queue.get()) != "":
            try:
                self._run(run_id)
            except Exception as e:  # never let one run stop the worker
                logger.exception("Training run %s failed", run_id)
                try:
                    run = self.store.get(run_id)
                    run.status, run.error, run.finished_at = "failed", str(e) or type(e).__name__, utcnow()
                    self.store.save(run)
                except KeyError:
                    pass
            finally:
                self.current, self._process = None, None

    def _run(self, run_id: str) -> None:
        try:
            run = self.store.get(run_id)
        except KeyError:
            return  # deleted while queued
        if run.status != "queued":
            return  # stopped while queued
        folder = self.store.run_dir(run_id)
        request = run.request
        work = self.work_dir / run_id
        work.parent.mkdir(parents=True, exist_ok=True)
        worker_request = {
            "base": self.base(request.base_model), "clips": self.clip_inputs(request.data.clip_ids),
            "data": request.data.model_dump(exclude={"clip_ids"}), "tokenizer": request.tokenizer.model_dump(),
            "processor": request.processor.model_dump(), "collator": request.collator.model_dump(),
            "metric": request.metric.model_dump(), "args": request.args.model_dump(), "threads": self.threads,
            "work_dir": str(work),
            "export": {"enabled": request.export, "model_id": run.model_id, "script": str(self.export_script),
                       "output": str(self.store.models_dir / run.name)},
        }
        (folder / "request.json").write_text(json.dumps(worker_request, indent=1))
        run.status, run.started_at, run.message = "running", utcnow(), "Starting the training Python"
        self.store.save(run)
        with (folder / "worker.log").open("ab") as log:
            self.current = run_id
            self._process = subprocess.Popen([self.python, "-m", "app.training_worker", "train", str(folder)],
                                             stdout=log, stderr=subprocess.STDOUT, env=self.env(), cwd=ROOT)
            code = self._process.wait()
        shutil.rmtree(work, ignore_errors=True)  # a killed worker leaves its checkpoints
        self._finish(run_id, code)

    def _finish(self, run_id: str, code: int) -> None:
        try:
            run = self.store.get(run_id)  # with the worker's status
        except KeyError:
            return  # deleted while running
        status = self.store.status(run_id)
        run.finished_at = utcnow()
        if run.status == "failed":  # deleted or failed meanwhile
            pass
        elif self._stopping and code != 0:
            run.status, run.error = "failed", INTERRUPTED
        elif code == 0 and status.get("phase") == "done":
            run.status, run.error = "completed", None
            model_dir = status.get("model_dir")
            if status.get("model_id") and model_dir and Path(model_dir).is_dir():
                self.models.add(Path(model_dir))
                run.exported = True
        else:
            run.status = "failed"
            run.error = status.get("error") or (
                f"The training process ended with code {code}" + (" (killed: out of memory? Lower "
                "per_device_train_batch_size, freeze the encoder, or use a bigger machine)" if code in (-9, 137) else "")
                + ": see its log")
        self.store.save(run)

    def log_tail(self, run_id: str, lines: int = 200) -> str:
        path = self.store.run_dir(run_id) / "worker.log"
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
