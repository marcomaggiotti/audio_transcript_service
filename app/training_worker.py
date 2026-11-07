"""Fine-tune Whisper on the benchmark set. Run by the training Python (PyTorch + transformers), never by the API:

    python -m app.training_worker train RUN_DIR    one run: RUN_DIR/request.json in, RUN_DIR/status.json out
    python -m app.training_worker inspect          one look at a component: JSON on stdin, JSON on stdout

The parts are those of Hugging Face's fine-tuning recipe (huggingface.co/blog/fine-tune-whisper):
- WhisperTokenizer: a clip's correct text -> the token ids the model learns to write (the labels). For an English-only
  model: <|startoftranscript|><|notimestamps|> text <|endoftext|>, no language or task token.
- WhisperFeatureExtractor: 16 kHz audio -> an 80 x 3000 log-mel spectrogram (30 s, padded with silence). With the
  tokenizer it makes a WhisperProcessor.
- The data collator: one batch of examples -> tensors. The spectrograms all have 3000 frames already; the labels are
  padded to the longest with -100, which the loss ignores, and <|startoftranscript|> is cut when every row starts
  with it (the model puts it in front of the labels itself).
- The metric: word error rate with the benchmark's normalization (app/wer.py), or character error rate.
- Seq2SeqTrainer with Seq2SeqTrainingArguments; evaluated with generate() on the clips held out for testing.

A run: the clips are split into training and test clips (never trained on), the base model is scored on the test
clips, trained, scored again with its best checkpoint, saved, and exported to ONNX (scripts/export_whisper_onnx.py)
so the service can transcribe with it. The worker writes its progress to status.json (the API reads it) and stops
at the next step when RUN_DIR/stop appears (then it still scores, saves and exports what it learned).
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio import decode_audio
from app.wer import normalize, word_error_rate

SAMPLE_RATE = 16_000
EXAMPLES = 8  # test clips shown before and after training
STATUS_EVERY = 2.0  # seconds between status writes while steps go by


# ---------------------------------------------------------------- the parts

def model_source(base: dict) -> tuple[str, str]:
    """Where to load (the processor, the PyTorch weights) of the base model: its PyTorch copy next to the ONNX export
    (built with TRAINING=true, or a fine-tuned model), else the export's tokenizer and Hugging Face for the weights."""
    folder = Path(base.get("dir") or ".")
    torch_dir = folder / "torch"
    if (torch_dir / "config.json").exists():
        return str(torch_dir), str(torch_dir)
    processor = str(folder) if (folder / "tokenizer.json").exists() and (folder / "preprocessor_config.json").exists() \
        else base["id"]
    return processor, base["id"]


def load_processor(base: dict, tokenizer: dict):
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_source(base)[0])
    processor.tokenizer.set_prefix_tokens(predict_timestamps=bool(tokenizer.get("predict_timestamps")))
    return processor


def label_text(text: str, tokenizer: dict) -> str:
    """The text the model learns to write: as written (case, punctuation), or as the metric compares it."""
    return " ".join(normalize(text)) if tokenizer.get("text") == "normalized" else " ".join(text.split())


def read_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    return decode_audio(Path(path), max_seconds=max_seconds)


def features_of(processor, audio: np.ndarray, settings: dict) -> np.ndarray:
    out = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, do_normalize=bool(settings.get("do_normalize")))
    return out.input_features[0]


@dataclass
class Collator:
    """Hugging Face's DataCollatorSpeechSeq2SeqWithPadding, with its settings."""

    processor: object
    decoder_start_token_id: int
    strip_start_token: bool = True
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features],
                                                     return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt",
                                                    pad_to_multiple_of=self.pad_to_multiple_of)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.label_pad_token_id)
        if self.strip_start_token and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _char_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def error_rates(references: list[str], predictions: list[str]) -> dict:
    """Word and character error rate over all the pairs (errors / reference words or characters), normalized as the
    benchmark does: case, punctuation and bracketed notes don't count."""
    words = errors = chars = char_errors = 0
    for ref, hyp in zip(references, predictions):
        result = word_error_rate(ref, hyp)
        words += result.reference_words
        errors += result.errors
        r, h = " ".join(normalize(ref)), " ".join(normalize(hyp))
        chars += len(r)
        char_errors += _char_distance(r, h)
    return {"wer": round(errors / words, 4) if words else 0.0, "cer": round(char_errors / chars, 4) if chars else 0.0,
            "words": words, "word_errors": errors}


# ---------------------------------------------------------------- inspecting a part

def _tokens(tokenizer, ids: list[int]) -> list[dict]:
    special = set(tokenizer.all_special_ids)
    out = []
    for i in ids:
        if i < 0:
            out.append({"id": i, "token": None, "special": False})
        else:
            token = tokenizer.convert_ids_to_tokens(i)
            marked = token.startswith("<|") and token.endswith("|>")  # timestamps aren't "special" but are
            out.append({"id": i, "token": token, "special": i in special or marked})
    return out


def inspect_tokenizer(request: dict) -> dict:
    settings = request.get("tokenizer") or {}
    tokenizer = load_processor(request["base"], settings).tokenizer
    text = request.get("text") or ""
    target = label_text(text, settings)
    ids = tokenizer(target).input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    named = {"<|startoftranscript|>": None, "<|notimestamps|>": None, "<|endoftext|>": None, "<|startofprev|>": None}
    return {
        "class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "model_max_length": tokenizer.model_max_length,
        "attributes": {"language": tokenizer.language, "task": tokenizer.task,
                       "predict_timestamps": bool(settings.get("predict_timestamps")),
                       "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token,
                       "pad_token_id": tokenizer.pad_token_id},
        "special": {t: tokenizer.convert_tokens_to_ids(t) for t in named},
        "prefix": _tokens(tokenizer, tokenizer.prefix_tokens),
        "text": target, "ids": ids, "tokens": _tokens(tokenizer, ids),
        "decoded_with_special": tokenizer.decode(ids, skip_special_tokens=False), "decoded": decoded,
        "round_trip": decoded == target,
    }


def _mel_preview(features: np.ndarray, frames: int, columns: int = 160) -> list[list[float]]:
    """The frames that hold audio, averaged down to at most `columns` columns, low frequencies first."""
    used = features[:, :max(1, min(frames, features.shape[1]))]
    step = max(1, math.ceil(used.shape[1] / columns))
    cols = [used[:, i:i + step].mean(axis=1) for i in range(0, used.shape[1], step)]
    return np.round(np.stack(cols, axis=1), 3).tolist()


def inspect_processor(request: dict) -> dict:
    settings = request.get("processor") or {}
    processor = load_processor(request["base"], request.get("tokenizer") or {})
    fe = processor.feature_extractor
    clip = request["clip"]
    audio = read_audio(clip["audio"])
    seconds = len(audio) / SAMPLE_RATE
    features = features_of(processor, audio[:fe.n_samples], settings)
    labels = processor.tokenizer(label_text(clip.get("reference") or "", request.get("tokenizer") or {})).input_ids
    frames = int(min(seconds, fe.chunk_length) * SAMPLE_RATE / fe.hop_length)
    return {
        "class": type(processor).__name__, "feature_extractor": type(fe).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "attributes": {"feature_size": fe.feature_size, "sampling_rate": fe.sampling_rate, "hop_length": fe.hop_length,
                       "n_fft": fe.n_fft, "chunk_length": fe.chunk_length, "n_samples": fe.n_samples,
                       "nb_max_frames": fe.nb_max_frames, "padding_value": fe.padding_value,
                       "do_normalize": bool(settings.get("do_normalize"))},
        "clip": {"id": clip.get("id"), "name": clip.get("name"), "seconds": round(seconds, 2),
                 "samples": len(audio), "cut": seconds > fe.chunk_length},
        "input_features_shape": list(features.shape), "frames_with_audio": frames,
        "stats": {"min": round(float(features.min()), 3), "max": round(float(features.max()), 3),
                  "mean": round(float(features.mean()), 3)},
        "mel": _mel_preview(features, frames),
        "labels": len(labels), "label_tokens": _tokens(processor.tokenizer, labels[:32]),
    }


def inspect_collator(request: dict) -> dict:
    settings = request.get("collator") or {}
    tok_settings = request.get("tokenizer") or {}
    processor = load_processor(request["base"], tok_settings)
    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    collator = Collator(processor, start, strip_start_token=settings.get("strip_start_token", True),
                        pad_to_multiple_of=settings.get("pad_to_multiple_of"))
    features = []
    for clip in request["clips"]:
        audio = read_audio(clip["audio"], max_seconds=processor.feature_extractor.chunk_length)
        features.append({"input_features": features_of(processor, audio, request.get("processor") or {}),
                         "labels": processor.tokenizer(label_text(clip.get("reference") or "", tok_settings)).input_ids})
    batch = collator(features)
    labels = batch["labels"].tolist()
    stripped = len(labels[0]) < max(len(f["labels"]) for f in features) if labels else False
    return {
        "class": "DataCollatorSpeechSeq2SeqWithPadding",
        "attributes": {"decoder_start_token_id": start, "strip_start_token": collator.strip_start_token,
                       "pad_to_multiple_of": collator.pad_to_multiple_of, "label_pad_token_id": collator.label_pad_token_id},
        "input_features_shape": list(batch["input_features"].shape), "labels_shape": list(batch["labels"].shape),
        "start_token_cut": stripped,
        "rows": [{"id": clip.get("id"), "name": clip.get("name"), "tokens": _tokens(processor.tokenizer, row),
                  "length": sum(1 for i in row if i != -100), "padding": sum(1 for i in row if i == -100)}
                 for clip, row in zip(request["clips"], labels)],
    }


INSPECTORS = {"tokenizer": inspect_tokenizer, "processor": inspect_processor, "collator": inspect_collator}


# ---------------------------------------------------------------- a run

class Status:
    """status.json, written whole and atomically: the API reads it while the run goes on."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "status.json"
        self.data: dict = {"phase": "starting", "history": [], "error": None}
        self._last = 0.0

    def update(self, force: bool = True, **changes) -> None:
        self.data.update(changes)
        now = time.monotonic()
        if not force and now - self._last < STATUS_EVERY:
            return
        self._last = now
        self.data["updated_at"] = time.time()
        tmp = self.path.with_name("status.json.tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


def split(clips: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Shuffled with the seed; at least one clip for testing and one for training."""
    order = list(clips)
    random.Random(seed).shuffle(order)
    n_test = min(len(order) - 1, max(1, round(len(order) * test_size)))
    return order[n_test:], order[:n_test]


def prepare(clips: list[dict], processor, request: dict) -> tuple[list[dict], list[dict]]:
    """(examples, skipped clips with the reason)."""
    tok_settings, proc_settings = request.get("tokenizer") or {}, request.get("processor") or {}
    max_seconds = float(proc_settings.get("max_audio_seconds") or 30)
    max_labels = int(tok_settings.get("max_label_tokens") or 448)
    examples, skipped = [], []
    for clip in clips:
        text = label_text(clip.get("reference") or "", tok_settings)
        if not text:
            skipped.append({"id": clip["id"], "name": clip["name"], "reason": "no correct text"})
            continue
        try:
            audio = read_audio(clip["audio"])
        except Exception as e:  # a missing or unreadable file: say so, train on the others
            skipped.append({"id": clip["id"], "name": clip["name"], "reason": f"audio unreadable: {e}"})
            continue
        seconds = len(audio) / SAMPLE_RATE
        if seconds > max_seconds:
            skipped.append({"id": clip["id"], "name": clip["name"],
                            "reason": f"{seconds:.1f} s: longer than {max_seconds:g} s (Whisper hears 30 s at a time)"})
            continue
        labels = processor.tokenizer(text).input_ids
        if len(labels) > max_labels:
            skipped.append({"id": clip["id"], "name": clip["name"],
                            "reason": f"{len(labels)} tokens: more than {max_labels}"})
            continue
        examples.append({"id": clip["id"], "name": clip["name"], "reference": clip.get("reference") or "",
                         "seconds": round(seconds, 2), "input_features": features_of(processor, audio, proc_settings),
                         "labels": labels})
    return examples, skipped


def decode(tokenizer, ids) -> list[str]:
    """Token ids -> texts; -100 (padding of batches of different lengths) read as the pad token."""
    ids = np.asarray(ids)
    return tokenizer.batch_decode(np.where(ids < 0, tokenizer.pad_token_id, ids), skip_special_tokens=True)


def _summary(pred, tokenizer, examples: list[dict], metric: str) -> dict:
    texts = decode(tokenizer, pred.predictions)
    rates = error_rates([e["reference"] for e in examples], texts)
    return {**rates, "metric": metric, "loss": round(float(pred.metrics.get("test_loss", float("nan"))), 4),
            "examples": [{"id": e["id"], "name": e["name"], "reference": e["reference"], "prediction": t.strip(),
                          "wer": round(word_error_rate(e["reference"], t).wer, 4)}
                         for e, t in zip(examples[:EXAMPLES], texts[:EXAMPLES])]}


def train(run_dir: Path) -> int:
    request = json.loads((run_dir / "request.json").read_text())
    status = Status(run_dir)
    work = Path(request.get("work_dir") or tempfile.mkdtemp(prefix="whisper-training-"))
    try:
        return _train(run_dir, request, status, work)
    except Exception as e:  # the API shows the error; the traceback goes to worker.log
        traceback.print_exc()
        status.update(phase="failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)  # intermediate checkpoints: the model is saved elsewhere


def _train(run_dir: Path, request: dict, status: Status, work: Path) -> int:
    status.update(phase="loading", message="Loading PyTorch, transformers and the base model")
    if hasattr(os, "nice"):
        os.nice(10)  # transcription jobs keep their priority
    import torch
    import transformers
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, WhisperForConditionalGeneration

    threads = int(request.get("threads") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    cuda = torch.cuda.is_available()
    status.update(device=torch.cuda.get_device_name(0) if cuda else "cpu", threads=torch.get_num_threads(),
                  versions={"torch": torch.__version__, "transformers": transformers.__version__})
    args_in, tok_settings, col_settings = request["args"], request.get("tokenizer") or {}, request.get("collator") or {}
    metric = (request.get("metric") or {}).get("name", "wer")
    transformers.set_seed(int(args_in.get("seed", 42)))

    processor = load_processor(request["base"], tok_settings)
    model = WhisperForConditionalGeneration.from_pretrained(model_source(request["base"])[1])
    # English-only checkpoints force <|notimestamps|> in an old way; generate() builds the prompt itself.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args_in.get("freeze_encoder"):
        model.freeze_encoder()
    if args_in.get("gradient_checkpointing"):
        model.config.use_cache = False

    status.update(phase="preparing", message="Turning the clips into spectrograms and labels")
    data = request.get("data") or {}
    train_clips, test_clips = split(request["clips"], float(data.get("test_size", 0.2)), int(data.get("seed", 42)))
    train_set, skipped_train = prepare(train_clips, processor, request)
    test_set, skipped_test = prepare(test_clips, processor, request)
    status.update(data={"train": len(train_set), "test": len(test_set), "skipped": skipped_train + skipped_test,
                        "train_seconds": round(sum(e["seconds"] for e in train_set), 1),
                        "test_seconds": round(sum(e["seconds"] for e in test_set), 1),
                        "test_clips": [{"id": e["id"], "name": e["name"]} for e in test_set]})
    if not train_set or not test_set:
        raise ValueError("Not enough usable clips: at least one for training and one for testing are needed")

    def compute_metrics(pred) -> dict:
        rates = error_rates(decode(processor.tokenizer, pred.label_ids), decode(processor.tokenizer, pred.predictions))
        return {"wer": rates["wer"], "cer": rates["cer"]}

    eval_steps = min(int(args_in["eval_steps"]), int(args_in["max_steps"]))
    args = Seq2SeqTrainingArguments(
        output_dir=str(work / "checkpoints"),
        per_device_train_batch_size=int(args_in["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(args_in["gradient_accumulation_steps"]),
        learning_rate=float(args_in["learning_rate"]),
        warmup_steps=int(args_in["warmup_steps"]),
        max_steps=int(args_in["max_steps"]),
        lr_scheduler_type=args_in.get("lr_scheduler_type", "linear"),
        weight_decay=float(args_in.get("weight_decay", 0.0)),
        gradient_checkpointing=bool(args_in.get("gradient_checkpointing")),
        fp16=bool(args_in.get("fp16")) and cuda,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        logging_steps=int(args_in["logging_steps"]),
        per_device_eval_batch_size=int(args_in["per_device_eval_batch_size"]),
        predict_with_generate=True, generation_max_length=int(args_in["generation_max_length"]),
        load_best_model_at_end=bool(args_in.get("load_best_model_at_end", True)),
        metric_for_best_model=metric, greater_is_better=False,
        save_total_limit=2, save_only_model=True, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=cuda, disable_tqdm=True, seed=int(args_in.get("seed", 42)),
    )
    stop_file = run_dir / "stop"

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stop_file.exists():
                control.should_training_stop = True
                status.update(stopped_at=state.global_step, message=f"Stopped at step {state.global_step}")
            status.update(force=False, step=state.global_step, max_steps=state.max_steps,
                          epoch=round(state.epoch or 0, 3))

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            keep = {k: round(float(v), 6) for k, v in logs.items() if isinstance(v, (int, float))
                    and k in ("loss", "learning_rate", "grad_norm", "eval_loss", "eval_wer", "eval_cer", "epoch")}
            if keep.keys() & {"loss", "eval_loss", "eval_wer", "eval_cer"}:  # not the end-of-training summary
                status.data["history"].append({"step": state.global_step, **keep})
            status.update(step=state.global_step, max_steps=state.max_steps, epoch=round(state.epoch or 0, 3))

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=test_set,
                             data_collator=Collator(processor, model.config.decoder_start_token_id,
                                                    strip_start_token=col_settings.get("strip_start_token", True),
                                                    pad_to_multiple_of=col_settings.get("pad_to_multiple_of")),
                             compute_metrics=compute_metrics, processing_class=processor.feature_extractor,
                             callbacks=[Progress()])

    status.update(phase="baseline", message="Scoring the base model on the test clips")
    status.update(baseline=_summary(trainer.predict(test_set), processor.tokenizer, test_set, metric))
    status.update(phase="training", message="Training", step=0, max_steps=int(args_in["max_steps"]))
    started = time.monotonic()
    result = trainer.train()
    status.update(train_seconds=round(time.monotonic() - started, 1),
                  train_loss=round(float(result.training_loss), 4), step=trainer.state.global_step,
                  best_step=int(trainer.state.best_model_checkpoint.rsplit("-", 1)[-1])
                  if trainer.state.best_model_checkpoint else None)
    status.update(phase="final", message="Scoring the trained model on the test clips")
    status.update(final=_summary(trainer.predict(test_set), processor.tokenizer, test_set, metric))

    status.update(phase="saving", message="Saving the model")
    saved = run_dir / "model"  # kept here if the export fails
    shutil.rmtree(saved, ignore_errors=True)
    trainer.save_model(str(saved))
    processor.save_pretrained(str(saved))
    trainer.model.generation_config.save_pretrained(str(saved))
    export = request.get("export") or {}
    if not export.get("enabled"):
        status.update(phase="done", message="Trained (not exported)", model_dir=str(saved))
        return 0

    status.update(phase="exporting", message="Exporting to ONNX for the service")
    out = Path(export["output"])
    tmp = out.with_name(out.name + ".exporting")
    shutil.rmtree(tmp, ignore_errors=True)
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    t0 = time.monotonic()
    subprocess.run([sys.executable, export["script"], "--model", str(saved), "--output", str(tmp), "--prune",
                    "--no-variants"], check=True, env=env)
    (tmp / "model_id.txt").write_text(export["model_id"] + "\n")
    shutil.move(str(saved), str(tmp / "torch"))  # the PyTorch weights: a base for training it further
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    status.update(phase="done", message="Trained and exported", model_id=export["model_id"], model_dir=str(out),
                  export_seconds=round(time.monotonic() - t0, 1))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "train":
        return train(Path(argv[1]))
    if argv[:1] == ["inspect"]:
        request = json.loads(sys.stdin.read())
        try:
            result = INSPECTORS[request["kind"]](request)
        except Exception as e:
            traceback.print_exc()
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
            return 1
        print(json.dumps(result))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
