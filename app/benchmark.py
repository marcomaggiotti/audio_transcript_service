"""Accuracy benchmark: word error rate of Whisper models and decoding options on your own audio.

The benchmark set is a list of reference clips. Each one is an interval of a real recording with its
correct transcript: upload a recording, let Whisper propose segments, pick the intervals that
represent the audio well, fix their text and save them. A run transcribes every clip with each
chosen model and decoding variant and scores the result against the references.

On disk (BENCHMARK_DIR):
    sources/<id>, sources/<id>.json     uploaded recordings and their proposed segments
    clips/<id>.wav, clips/<id>.txt      interval audio (16 kHz mono) and its correct transcript
    clips/<id>.json                     name, source recording, start/end
    runs/<id>.json                      finished runs with their results
The clips directory is a plain "audio + same-name .txt" folder, so scripts/wer.py can score it too.
"""

import gc
import json
import logging
import re
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.asr import DecodingOptions, Transcriber
from app.audio import SAMPLE_RATE, InvalidAudioError, decode_audio, duration_seconds
from app.metrics import peak_memory_mb, reset_peak_memory
from app.schemas import (
    BenchmarkModel,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSource,
    BenchmarkVariant,
    ClipResult,
    JobStatus,
    ReferenceClipInfo,
    utcnow,
)
from app.wer import word_error_rate

logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"[0-9a-f]{32}")
MIN_CLIP_SECONDS = 0.5
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".flac")

VARIANTS = {
    v.id: v for v in (
        BenchmarkVariant(id="greedy", label="Greedy", temperature_fallback=False, condition_on_previous_text=False),
        BenchmarkVariant(id="fallback", label="Temperature fallback", temperature_fallback=True,
                         condition_on_previous_text=False),
        BenchmarkVariant(id="fallback_previous", label="Fallback + previous text", temperature_fallback=True,
                         condition_on_previous_text=True),
    )
}


class BenchmarkError(ValueError):
    """A request the benchmark can't carry out (bad interval, unknown model or clip)."""


class NotFound(KeyError):
    pass


def variant_options(variant: str, initial_prompt: str | None = None) -> DecodingOptions:
    v = VARIANTS[variant]
    return DecodingOptions(temperature_fallback=v.temperature_fallback,
                           condition_on_previous_text=v.condition_on_previous_text,
                           initial_prompt=initial_prompt or None)


# ------------------------------------------------------------------ scoring


@dataclass
class EvalClip:
    id: str
    name: str
    path: Path
    reference: str


def clips_from_dir(directory: Path) -> list[EvalClip]:
    """Audio files that have a same-name .txt next to them holding the correct transcript."""
    clips = []
    for audio in sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS):
        reference = audio.with_suffix(".txt")
        if reference.exists():
            meta = audio.with_suffix(".json")
            name = json.loads(meta.read_text()).get("name", audio.stem) if meta.exists() else audio.stem
            clips.append(EvalClip(audio.stem, name, audio, reference.read_text().strip()))
    return clips


def evaluate(transcriber: Transcriber, clips: list[EvalClip], language: str | None, options: DecodingOptions,
             on_clip: Callable[[int, EvalClip], None] | None = None) -> BenchmarkResult:
    """Transcribe every clip and score it. WER is over all clips together (errors / reference words).

    on_clip(i, clip) is called before the i-th clip.
    """
    result = BenchmarkResult(model="", variant="")
    reset_peak_memory()
    for i, clip in enumerate(clips):
        if on_clip:
            on_clip(i, clip)
        audio = decode_audio(clip.path, as_int16=True)
        t0 = time.perf_counter()
        hypothesis = transcriber.transcribe(audio, language=language, options=options).text
        seconds = time.perf_counter() - t0
        score = word_error_rate(clip.reference, hypothesis)
        result.substitutions += score.substitutions
        result.deletions += score.deletions
        result.insertions += score.insertions
        result.reference_words += score.reference_words
        result.audio_seconds += duration_seconds(audio)
        result.inference_seconds += seconds
        result.clips.append(ClipResult(
            clip_id=clip.id, name=clip.name, wer=round(score.wer, 4), errors=score.errors,
            reference_words=score.reference_words, audio_seconds=round(duration_seconds(audio), 2),
            inference_seconds=round(seconds, 3), hypothesis=hypothesis, alignment=score.alignment,
        ))
    errors = result.substitutions + result.deletions + result.insertions
    result.wer = round(errors / result.reference_words, 4) if result.reference_words else None
    result.audio_seconds = round(result.audio_seconds, 2)
    result.inference_seconds = round(result.inference_seconds, 3)
    result.real_time_factor = (round(result.inference_seconds / result.audio_seconds, 3)
                               if result.audio_seconds else None)
    result.peak_memory_mb = peak_memory_mb()
    return result


# ------------------------------------------------------------------- storage


def write_wav(path: Path, samples: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.astype(np.int16).tobytes())
    tmp.replace(path)


class BenchmarkStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.sources_dir = self.root / "sources"
        self.clips_dir = self.root / "clips"
        self.runs_dir = self.root / "runs"
        self._lock = threading.Lock()

    def setup(self) -> None:
        for directory in (self.sources_dir, self.clips_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_id(item_id: str) -> str:
        if not ID_PATTERN.fullmatch(item_id):
            raise NotFound(item_id)  # also keeps ids from naming paths outside the store
        return item_id

    def _write_json(self, path: Path, model) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(model.model_dump_json(indent=1))
        tmp.replace(path)

    # ----------------------------------------------------------- sources

    def source_audio(self, source_id: str) -> Path:
        return self.sources_dir / self._check_id(source_id)

    def save_source(self, source: BenchmarkSource) -> None:
        with self._lock:
            self._write_json(self.sources_dir / f"{self._check_id(source.id)}.json", source)

    def get_source(self, source_id: str) -> BenchmarkSource:
        path = self.sources_dir / f"{self._check_id(source_id)}.json"
        if not path.exists():
            raise NotFound(source_id)
        return BenchmarkSource.model_validate_json(path.read_text())

    def list_sources(self) -> list[BenchmarkSource]:
        sources = [BenchmarkSource.model_validate_json(p.read_text()) for p in self.sources_dir.glob("*.json")]
        return sorted(sources, key=lambda s: s.created_at, reverse=True)

    def delete_source(self, source_id: str) -> None:
        """Remove a recording. Clips cut from it keep their own audio."""
        self.get_source(source_id)
        (self.sources_dir / f"{source_id}.json").unlink(missing_ok=True)
        self.source_audio(source_id).unlink(missing_ok=True)

    # ------------------------------------------------------------- clips

    def clip_audio(self, clip_id: str) -> Path:
        return self.clips_dir / f"{self._check_id(clip_id)}.wav"

    def add_clip(self, info: ReferenceClipInfo, samples: np.ndarray) -> ReferenceClipInfo:
        self._check_id(info.id)
        write_wav(self.clip_audio(info.id), samples)
        self._save_clip(info)
        return info

    def _save_clip(self, info: ReferenceClipInfo) -> None:
        with self._lock:
            (self.clips_dir / f"{info.id}.txt").write_text(info.reference.strip() + "\n")
            self._write_json(self.clips_dir / f"{info.id}.json", info.model_copy(update={"reference": ""}))

    def get_clip(self, clip_id: str) -> ReferenceClipInfo:
        meta = self.clips_dir / f"{self._check_id(clip_id)}.json"
        if not meta.exists():
            raise NotFound(clip_id)
        info = ReferenceClipInfo.model_validate_json(meta.read_text())
        info.reference = (self.clips_dir / f"{clip_id}.txt").read_text().strip()  # the .txt is the reference
        return info

    def list_clips(self) -> list[ReferenceClipInfo]:
        clips = [self.get_clip(p.stem) for p in self.clips_dir.glob("*.json") if ID_PATTERN.fullmatch(p.stem)]
        return sorted(clips, key=lambda c: c.created_at)

    def update_clip(self, clip_id: str, reference: str | None = None, name: str | None = None) -> ReferenceClipInfo:
        info = self.get_clip(clip_id)
        if reference is not None:
            info.reference = reference.strip()
        if name is not None:
            info.name = name.strip()
        self._save_clip(info)
        return info

    def delete_clip(self, clip_id: str) -> None:
        self.get_clip(clip_id)
        for suffix in (".wav", ".txt", ".json"):
            (self.clips_dir / f"{clip_id}{suffix}").unlink(missing_ok=True)

    def eval_clips(self, clip_ids: list[str] | None = None) -> list[EvalClip]:
        infos = self.list_clips() if clip_ids is None else [self.get_clip(c) for c in clip_ids]
        return [EvalClip(c.id, c.name, self.clip_audio(c.id), c.reference) for c in infos]

    # -------------------------------------------------------------- runs

    def save_run(self, run: BenchmarkRun) -> None:
        with self._lock:
            self._write_json(self.runs_dir / f"{self._check_id(run.id)}.json", run)

    def get_run(self, run_id: str) -> BenchmarkRun:
        path = self.runs_dir / f"{self._check_id(run_id)}.json"
        if not path.exists():
            raise NotFound(run_id)
        return BenchmarkRun.model_validate_json(path.read_text())

    def list_runs(self) -> list[BenchmarkRun]:
        runs = [BenchmarkRun.model_validate_json(p.read_text()) for p in self.runs_dir.glob("*.json")]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def delete_run(self, run_id: str) -> None:
        self.get_run(run_id)
        (self.runs_dir / f"{run_id}.json").unlink(missing_ok=True)


def cut_interval(path: Path, start: float, end: float, max_seconds: float) -> np.ndarray:
    """Decode [start, end) seconds of a recording as 16 kHz mono int16."""
    if end - start < MIN_CLIP_SECONDS:
        raise BenchmarkError(f"An interval must be at least {MIN_CLIP_SECONDS}s long")
    if end - start > max_seconds:
        raise BenchmarkError(f"An interval can be at most {max_seconds:.0f}s long")
    audio = decode_audio(path, max_seconds=end, as_int16=True)
    if start >= duration_seconds(audio) - MIN_CLIP_SECONDS / 2:
        raise BenchmarkError(f"The interval starts after the end of the recording ({duration_seconds(audio):.1f}s)")
    return audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)].copy()


# --------------------------------------------------------------- the runner


def discover_models(models_dir: str) -> dict[str, Path]:
    """Subdirectories holding a Whisper export (scripts/export_whisper_onnx.py), by directory name."""
    root = Path(models_dir)
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "encoder_model.onnx").exists() and (p / "tokenizer.json").exists()}


def model_name(path: Path) -> str:
    model_id = path / "model_id.txt"  # written by scripts/export_whisper_onnx.py
    return model_id.read_text().strip() if model_id.exists() else path.name


class BenchmarkRunner:
    """Queues runs on the transcription workers (so a run never overlaps a job) and keeps their results."""

    def __init__(self, store: BenchmarkStore, submit: Callable[[str, Callable[[], None]], None],
                 default_model: tuple[str, Transcriber] | None, models_dir: str,
                 load_model: Callable[[Path], Transcriber]):
        self.store = store
        self.submit = submit  # TranscriptionService.submit_task
        self.default_model = default_model  # (id, loaded transcriber) of the service's own model
        self.models_dir = models_dir
        self.load_model = load_model
        self.active: dict[str, BenchmarkRun] = {}  # queued or running, with live progress

    def recover(self) -> None:
        """Runs a restart interrupted can't resume: mark them failed."""
        for run in self.store.list_runs():
            if run.status in (JobStatus.queued, JobStatus.processing):
                run.status, run.error, run.finished_at = JobStatus.failed, "Interrupted by a restart", utcnow()
                self.store.save_run(run)

    def models(self) -> list[BenchmarkModel]:
        models = []
        if self.default_model:
            model_id, transcriber = self.default_model
            models.append(BenchmarkModel(id=model_id, name=transcriber.name, default=True))
        for model_id, path in discover_models(self.models_dir).items():
            if not self.default_model or model_id != self.default_model[0]:
                models.append(BenchmarkModel(id=model_id, name=model_name(path), default=False))
        return models

    def get(self, run_id: str) -> BenchmarkRun:
        return self.active.get(run_id) or self.store.get_run(run_id)

    def all_runs(self) -> list[BenchmarkRun]:
        stored = [r for r in self.store.list_runs() if r.id not in self.active]
        return sorted([*self.active.values(), *stored], key=lambda r: r.created_at, reverse=True)

    def create(self, run: BenchmarkRun) -> BenchmarkRun:
        known = {m.id for m in self.models()}
        if unknown := [m for m in run.models if m not in known]:
            raise BenchmarkError(f"Unknown model(s): {', '.join(unknown)}; see GET /benchmark/models")
        if unknown := [v for v in run.variants if v not in VARIANTS]:
            raise BenchmarkError(f"Unknown variant(s): {', '.join(unknown)}; choose from {', '.join(VARIANTS)}")
        try:
            clips = self.store.eval_clips(run.clip_ids or None)
        except NotFound as e:
            raise BenchmarkError(f"Unknown clip {e.args[0]}") from None
        if not clips:
            raise BenchmarkError("No reference clips yet: add some first")
        run.clip_ids = [c.id for c in clips]
        run.progress.total = len(run.models) * len(run.variants) * len(clips)
        self.store.save_run(run)
        self.active[run.id] = run
        try:
            self.submit(run.id, lambda: self._run(run, clips))
        except Exception:
            del self.active[run.id]
            self.store.delete_run(run.id)
            raise
        return run

    def _run(self, run: BenchmarkRun, clips: list[EvalClip]) -> None:
        run.status, run.started_at = JobStatus.processing, utcnow()
        language = None if run.language == "auto" else run.language
        try:
            for model_id in dict.fromkeys(run.models):
                self._run_model(run, model_id, clips, language)
            run.status = JobStatus.completed
        except Exception as e:  # results so far are kept
            logger.exception("Benchmark %s failed", run.id)
            run.status, run.error = JobStatus.failed, str(e) or e.__class__.__name__
        finally:
            run.progress.current = None
            run.finished_at = utcnow()
            self.store.save_run(run)
            self.active.pop(run.id, None)

    def _run_model(self, run: BenchmarkRun, model_id: str, clips: list[EvalClip], language: str | None) -> None:
        transcriber, load_seconds, error = None, None, None
        if self.default_model and model_id == self.default_model[0]:
            transcriber = self.default_model[1]
        else:
            t0 = time.perf_counter()
            run.progress.current = f"loading {model_id}"
            try:
                transcriber = self.load_model(discover_models(self.models_dir)[model_id])
                load_seconds = round(time.perf_counter() - t0, 2)
            except Exception as e:  # e.g. an export that doesn't load: report it, carry on with the others
                error = f"Model didn't load: {e}"
        try:
            for variant in run.variants:
                if transcriber is None:
                    result = BenchmarkResult(model=model_id, variant=variant, error=error)
                    run.progress.done += len(clips)
                else:
                    done = run.progress.done

                    def on_clip(i: int, clip: EvalClip, variant=variant) -> None:
                        run.progress.done = done + i
                        run.progress.current = f"{model_id} · {VARIANTS[variant].label} · {clip.name}"

                    try:
                        result = evaluate(transcriber, clips, language, variant_options(variant, run.initial_prompt),
                                          on_clip=on_clip)
                    except (InvalidAudioError, ValueError) as e:
                        result = BenchmarkResult(model=model_id, variant=variant, error=str(e))
                    run.progress.done = done + len(clips)
                    result.model, result.variant = model_id, variant
                    result.model_name, result.load_seconds = transcriber.name, load_seconds
                run.results.append(result)
                self.store.save_run(run)
        finally:
            if transcriber is not None and not (self.default_model and transcriber is self.default_model[1]):
                del transcriber
                gc.collect()  # free the extra model before the next one loads
