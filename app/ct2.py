"""CTranslate2: the same Whisper checkpoint on another inference engine, through faster-whisper.

The export (scripts/export_whisper_onnx.py --ct2) converts the service's checkpoint with ct2-transformers-converter
into WHISPER_MODEL_DIR/ct2 (float16 weights, its tokenizer and preprocessor config), so both engines run the same
model and a job can compare them (POST /jobs engine=ctranslate2). The weights are converted again on load to the
job's compute type: int8 (the usual CPU choice: smallest and fastest), int8_float32 or float32.

faster-whisper does its own windowing, timestamps, beam search, temperature fallback and VAD (its bundled Silero),
with the job's options. The model is loaded for the job and released after it, like another model (the service's
ONNX model stays loaded); its packages are imported only then.
"""

import importlib.util
import logging
from pathlib import Path

import numpy as np

from app.asr import FALLBACK_TEMPERATURES, DecodingOptions, Segment, TranscriptionResult, _release_memory

logger = logging.getLogger(__name__)

COMPUTE_TYPES = ("int8", "int8_float32", "float32")
WEIGHTS_FACTOR = 1.3  # resident memory of the loaded weights, relative to the converted file (float16 -> int8 ~0.6)


class CT2Unavailable(RuntimeError):
    pass


class CT2Engine:
    def __init__(self, model_dir: str | Path | None, threads: int = 0):
        self.path = Path(model_dir) if model_dir else None
        self.threads = threads

    @property
    def name(self) -> str:
        model_id = self.path / "model_id.txt" if self.path else None
        return model_id.read_text().strip() if model_id and model_id.exists() else (self.path.parent.name if self.path else "?")

    def error(self) -> str | None:
        """Why a job can't use it (None: it can)."""
        if self.path is None or not (self.path / "model.bin").exists():
            return (f"no CTranslate2 model at {self.path}: export with scripts/export_whisper_onnx.py --ct2"
                    if self.path else "CT2_MODEL_DIR isn't set")
        missing = [m for m in ("ctranslate2", "faster_whisper") if importlib.util.find_spec(m) is None]
        if missing:
            return f"{' and '.join(missing)} not installed"
        return None

    def size_mb(self) -> float | None:
        model = self.path / "model.bin" if self.path else None
        return round(model.stat().st_size / 1e6, 1) if model and model.exists() else None

    def status(self) -> dict:
        error = self.error()
        return {"available": error is None, "error": error, "model": self.name if error is None else None,
                "compute_types": list(COMPUTE_TYPES), "default_compute_type": "int8", "size_mb": self.size_mb(),
                "runtime": "CTranslate2 (CPU) through faster-whisper"}

    def needs_mb(self) -> float:
        return round((self.size_mb() or 0) * WEIGHTS_FACTOR + 150)

    def load(self, compute_type: str | None = None):
        """The model, its weights converted to `compute_type` (default int8)."""
        error = self.error()
        if error:
            raise CT2Unavailable(f"CTranslate2 is unavailable on this service: {error}")
        from faster_whisper import WhisperModel

        return WhisperModel(str(self.path), device="cpu", compute_type=compute_type or "int8", cpu_threads=self.threads,
                            local_files_only=True)

    def transcribe(self, model, audio: np.ndarray, language: str | None = None, options: DecodingOptions | None = None,
                   on_progress=None) -> TranscriptionResult:
        """Transcribe 16 kHz mono audio (int16 or float32) with the job's decoding options."""
        options = options or DecodingOptions()
        samples = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
        duration = samples.shape[0] / 16000 or 1.0
        vad = {k: v for k, v in (("threshold", options.vad_threshold),
                                  ("min_silence_duration_ms", options.vad_min_silence_ms)) if v is not None} or None
        try:
            pieces, info = model.transcribe(
                samples, language=language, task="transcribe", beam_size=max(1, options.beam_size or 1),
                temperature=list(FALLBACK_TEMPERATURES) if options.temperature_fallback else 0.0,
                condition_on_previous_text=bool(options.condition_on_previous_text),
                initial_prompt=options.initial_prompt or None, vad_filter=bool(options.vad), vad_parameters=vad)
            segments = []
            for s in pieces:  # a generator: the decoding happens here, window by window
                segments.append(Segment(start=round(float(s.start), 2), end=round(float(s.end), 2), text=s.text.strip(),
                                        avg_logprob=getattr(s, "avg_logprob", None),
                                        no_speech_prob=getattr(s, "no_speech_prob", None)))
                if on_progress:
                    on_progress(min(1.0, float(s.end) / duration))
        finally:
            del model  # the caller passes its only reference: the weights are freed here
            _release_memory()
        return TranscriptionResult(language=getattr(info, "language", None) or language, segments=segments)
