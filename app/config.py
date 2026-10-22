"""Runtime configuration, read from environment variables."""

import os
from functools import lru_cache

from pydantic import BaseModel


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _flag(name: str, default: str = "false") -> bool:
    return (_env(name, default) or "").lower() in ("1", "true", "yes")


class Settings(BaseModel):
    # HTTP
    host: str = "0.0.0.0"
    port: int = int(_env("PORT", "8000"))

    # Speech recognition (Whisper exported to ONNX, see scripts/export_whisper_onnx.py)
    whisper_model_dir: str = _env("WHISPER_MODEL_DIR", "./models/whisper-tiny")
    # Used when a request doesn't specify one; "auto" detects the language from the first window.
    default_language: str = _env("DEFAULT_LANGUAGE", "auto")
    max_new_tokens: int = int(_env("MAX_NEW_TOKENS", "224"))
    onnx_threads: int = int(_env("ONNX_THREADS", "0"))  # 0 = let ONNX Runtime decide
    # Skip ONNX Runtime's pre-packed weight copies: ~25 MB less RAM, decoder steps ~2x slower.
    whisper_low_memory: bool = _flag("WHISPER_LOW_MEMORY")
    # Decoding defaults (a request can override them): openai/whisper's temperature fallback, and
    # prompting each 30 s window with the text before it. INITIAL_PROMPT: vocabulary/style hint.
    temperature_fallback: bool = _flag("TEMPERATURE_FALLBACK", "true")
    condition_on_previous_text: bool = _flag("CONDITION_ON_PREVIOUS_TEXT", "true")
    initial_prompt: str | None = _env("INITIAL_PROMPT")

    # Benchmark tab: every subdirectory of MODELS_DIR holding an export can be compared.
    models_dir: str = _env("MODELS_DIR", "./models")
    # Reference recordings, clips (interval audio + correct transcript) and finished runs.
    benchmark_dir: str = _env("BENCHMARK_DIR", "./data/benchmark")
    max_clip_seconds: int = int(_env("MAX_CLIP_SECONDS", "300"))

    # Shared secret for the transcription endpoints. Empty = no auth.
    api_token: str | None = _env("API_TOKEN")

    # Uploads: kept in this directory only until their job has been transcribed.
    upload_dir: str = _env("UPLOAD_DIR", "./data/uploads")
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "50"))
    max_audio_seconds: int = int(_env("MAX_AUDIO_SECONDS", "1800"))

    # Background worker
    worker_concurrency: int = int(_env("WORKER_CONCURRENCY", "1"))
    max_queue_size: int = int(_env("MAX_QUEUE_SIZE", "20"))
    # Finished jobs kept in memory for GET /jobs/{id}; the oldest are forgotten beyond this.
    max_finished_jobs: int = int(_env("MAX_FINISHED_JOBS", "200"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
