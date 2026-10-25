"""The speech models a job can choose: the service's own (loaded at start) and the extra exports in MODELS_DIR.

The service's model stays loaded. Another model is loaded for the job that asks for it and released after
it, like a non-default precision: jobs run one at a time, so at most one extra model is in memory. Before
loading one, its size is checked against the container's memory, so a model too big for the plan is
refused with the reason instead of the process being killed (whisper-small needs ~1 GB: Render Standard).
"""

import gc
import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.asr import PRECISIONS, Transcriber, _release_memory, onnx_precision

logger = logging.getLogger(__name__)

RUNTIME_OVERHEAD_MB = 150  # ONNX Runtime's working memory, the audio and its features, for one job
WEIGHTS_FACTOR = 1.3  # resident memory of a loaded ONNX graph, relative to its file size
HEADROOM = 0.9  # never plan to use the last 10% of the container's memory


class ModelUnavailable(ValueError):
    """Unknown model, or one that doesn't fit in this deployment's memory."""


@dataclass
class ModelInfo:
    id: str  # the Hugging Face id from model_id.txt, e.g. openai/whisper-small
    path: Path
    default: bool = False
    precision: dict[str, str] = field(default_factory=dict)  # the export's default {encoder, decoder}
    available_precisions: dict[str, list[str]] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    size_mb: float = 0.0  # the default encoder + decoder files

    @property
    def needs_mb(self) -> float:
        """Memory the model needs loaded and transcribing, beyond what the process already uses."""
        return round(self.size_mb * WEIGHTS_FACTOR + RUNTIME_OVERHEAD_MB)

    def view(self, fits: bool | None, reason: str | None) -> dict:
        return {"id": self.id, "default": self.default, "precision": self.precision,
                "available_precisions": self.available_precisions, "languages": len(self.languages),
                "multilingual": len(self.languages) > 1, "size_mb": round(self.size_mb),
                "needs_mb": None if self.default else self.needs_mb, "fits": fits, "reason": reason}


def describe_export(path: Path, default: bool = False) -> ModelInfo:
    """What an export folder holds, from its files alone (nothing is loaded)."""
    decoder = next((n for n in ("decoder_model_merged.onnx", "decoder_model.onnx") if (path / n).exists()),
                   "decoder_model_merged.onnx")
    model_id = path / "model_id.txt"
    info = ModelInfo(id=model_id.read_text().strip() if model_id.exists() else path.name, path=path, default=default)
    for part, default_file in (("encoder", "encoder_model.onnx"), ("decoder", decoder)):
        main = path / default_file
        try:
            info.precision[part] = onnx_precision(main)
        except (OSError, ValueError):  # missing or empty (a placeholder): nothing to say about this part
            continue
        info.size_mb += main.stat().st_size / 1e6
        stem = default_file.removesuffix(".onnx")
        info.available_precisions[part] = sorted({info.precision[part]} | {
            p for p in PRECISIONS if (path / f"{stem}.{p}.onnx").exists()})
    try:
        config = json.loads((path / "generation_config.json").read_text())
        info.languages = [code.strip("<|>") for code in (config.get("lang_to_id") or {})] or ["en"]
    except (OSError, ValueError):
        info.languages = []
    return info


def memory_limit_mb() -> float | None:
    """The container's memory limit (cgroup v2 or v1), or MEMORY_LIMIT_MB; None if unlimited/unknown."""
    if os.getenv("MEMORY_LIMIT_MB"):
        return float(os.environ["MEMORY_LIMIT_MB"])
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit() and int(value) < 1 << 60:  # v1 reports "no limit" as a huge number
            return int(value) / 1024 / 1024
        return None
    return None


def current_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def discover_exports(models_dir: str) -> list[Path]:
    root = Path(models_dir)
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "encoder_model.onnx").exists() and (p / "tokenizer.json").exists()]


class ModelSet:
    def __init__(self, default: Transcriber | None, default_dir: str | None, models_dir: str | None,
                 load: Callable[[Path], Transcriber] | None, limit_mb: float | None = None,
                 rss_mb: Callable[[], float] = current_rss_mb):
        self.default = default
        self.load = load
        self.limit_mb = limit_mb
        self.rss_mb = rss_mb
        default_path = Path(default_dir).resolve() if default_dir else None
        self._models: dict[str, ModelInfo] = {}
        if default is not None:
            info = describe_export(default_path, default=True) if default_path and default_path.is_dir() \
                else ModelInfo(id=default.name, path=default_path or Path("."), default=True)
            info.id = default.name  # what jobs report as their model
            self._models[info.id] = info
        for path in discover_exports(models_dir) if models_dir else []:
            if default_path and path.resolve() == default_path:
                continue
            info = describe_export(path)
            self._models.setdefault(info.id, info)  # the same model exported twice: the first wins

    def list(self) -> list[ModelInfo]:
        return list(self._models.values())

    def find(self, model_id: str | None) -> ModelInfo | None:
        """By id (openai/whisper-small), short name (whisper-small) or folder name; None = the default."""
        if not model_id:
            return next((m for m in self._models.values() if m.default), None)
        for m in self._models.values():
            if model_id in (m.id, m.id.rsplit("/", 1)[-1], m.path.name):
                return m
        return None

    def fits(self, info: ModelInfo) -> tuple[bool | None, str | None]:
        """Whether it can be loaded next to what the process already holds (None: no limit known)."""
        if info.default:
            return True, None
        if self.limit_mb is None:
            return None, None
        free = self.limit_mb * HEADROOM - self.rss_mb()
        if info.needs_mb <= free:
            return True, None
        return False, (f"{info.id} needs about {info.needs_mb:.0f} MB and this server has about {max(free, 0):.0f} MB "
                       f"free of {self.limit_mb:.0f} MB: use a larger plan (Render Standard, 2 GB, for whisper-small)")

    def check(self, model_id: str | None, language: str | None = None) -> ModelInfo:
        info = self.find(model_id)
        if info is None:
            have = ", ".join(m.id for m in self._models.values()) or "none"
            raise ModelUnavailable(f"Unknown model '{model_id}' (this service has: {have}); see GET /config")
        if language and info.languages and language not in info.languages:
            raise ModelUnavailable(f"{info.id} can't transcribe '{language}'"
                                   + (" (English-only model)" if info.languages == ["en"] else ""))
        ok, reason = self.fits(info)
        if ok is False:
            raise ModelUnavailable(reason)
        return info

    @contextmanager
    def use(self, model_id: str | None, on_load: Callable[[str], None] | None = None) -> Iterator[Transcriber]:
        """The transcriber for a job: the loaded default, or another model loaded for it and released after."""
        info = self.check(model_id)
        if info.default:
            yield self.default
            return
        if on_load:
            on_load(info.id)
        logger.info("Loading %s for a job", info.id)
        transcriber = self.load(info.path)
        try:
            yield transcriber
        finally:
            del transcriber
            gc.collect()
            _release_memory()  # hand the extra model's weights back to the system before the next job
