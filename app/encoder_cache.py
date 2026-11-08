"""Encoder outputs kept and reused, and encoder sessions shared between models (app/asr.py).

Whisper's encoder turns each 30 s window of audio into 1500 vectors: the largest single step per window, the same for
every window (a model with whisper-tiny.en's shapes on one CPU thread: 0.29 s, of 0.77-1.2 s per window with 50-150
tokens decoded greedily). The same window through the same encoder always gives the same vectors, so they are kept and
reused, keyed by
- the encoder: a digest of its ONNX file (with its external data) and the graph optimization level it was opened at;
- the window: a digest of its log-mel spectrogram, exactly what the encoder reads.
A benchmark clip run again with other decoding options (beam search, fallback, a prompt), the same lines or VAD pieces
transcribed again, windows of digital silence, and a model fine-tuned with its encoder frozen (its encoder file is its
base model's, so the key is the same) all reuse them.

In memory: least recently used first out, at most ENCODER_CACHE_MB in all (0 = off). With ENCODER_CACHE_DIR, also on
disk, at most ENCODER_CACHE_DISK_MB (the least recently used deleted first): kept across restarts and read back instead
of encoded. Exact: the float32 arrays as the encoder computed them, so a transcript is the same with or without it.

Encoder sessions are shared the same way: a model whose encoder file is, byte for byte, the one of a loaded model gets
that session instead of a second copy in memory (a fine-tuned model and its base, when the encoder was frozen).

The decoder's KV cache can't be kept like this: its keys and values depend on the window's audio (every decoder layer
after the first attends to it) and on the text written so far, so no two windows share them.
"""

import hashlib
import logging
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DIGESTS: dict[tuple[str, int, int], str] = {}
_DIGEST_LOCK = threading.Lock()


def file_digest(path: Path) -> str:
    """A digest of an ONNX file's content and its external data (computed once while their size and mtime stay)."""
    path = Path(path)
    data = path.with_name(path.name + "_data")
    parts = [path, data] if data.exists() else [path]
    stats = [p.stat() for p in parts]
    key = (str(path.resolve()), sum(s.st_size for s in stats), max(s.st_mtime_ns for s in stats))
    with _DIGEST_LOCK:
        if key in _DIGESTS:
            return _DIGESTS[key]
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        with p.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    digest = h.hexdigest()
    with _DIGEST_LOCK:
        _DIGESTS[key] = digest
    return digest


def mel_digest(mel: np.ndarray) -> str:
    """A digest of one window's log-mel spectrogram (its values and shape)."""
    data = np.ascontiguousarray(mel, dtype=np.float32)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(data.shape).encode())
    h.update(memoryview(data).cast("B"))
    return h.hexdigest()


class EncoderCache:
    """Encoder outputs by key: in memory (LRU, bounded in bytes) and, optionally, on disk (bounded, kept)."""

    def __init__(self, limit_mb: float = 64, disk_dir: str | Path | None = None, disk_limit_mb: float = 0):
        self.limit = max(0, int(limit_mb * 1e6))
        self.disk = Path(disk_dir) if disk_dir and disk_limit_mb > 0 else None
        self.disk_limit = max(0, int(disk_limit_mb * 1e6))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._disk_bytes = 0
        self._lock = threading.Lock()
        self.hits = self.disk_hits = self.misses = 0
        if self.disk is not None:
            try:
                self.disk.mkdir(parents=True, exist_ok=True)
                for leftover in self.disk.glob("*.tmp"):  # a write cut off by a restart
                    leftover.unlink(missing_ok=True)
                self._disk_bytes = sum(p.stat().st_size for p in self.disk.glob("*.npy"))
            except OSError as e:
                logger.warning("Encoder cache on disk unavailable (%s): memory only", e)
                self.disk = None

    @property
    def enabled(self) -> bool:
        return self.limit > 0 or self.disk is not None

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                self.hits += 1
                return found
        if self.disk is not None:
            path = self.disk / f"{key}.npy"
            try:
                found = np.load(path, allow_pickle=False)
                os.utime(path)  # recently used: deleted last
            except (OSError, ValueError):
                found = None
            if found is not None:
                found.setflags(write=False)
                self._remember(key, found)
                with self._lock:
                    self.hits += 1
                    self.disk_hits += 1
                return found
        with self._lock:
            self.misses += 1
        return None

    def put(self, key: str, value: np.ndarray) -> np.ndarray:
        """Keep it (read-only from now on) and return it."""
        value = np.ascontiguousarray(value, dtype=np.float32)
        value.setflags(write=False)
        self._remember(key, value)
        if self.disk is not None:
            self._store(key, value)
        return value

    def _remember(self, key: str, value: np.ndarray) -> None:
        if value.nbytes > self.limit:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = value
            self._bytes += value.nbytes
            while self._bytes > self.limit and self._items:
                _, gone = self._items.popitem(last=False)
                self._bytes -= gone.nbytes

    def _store(self, key: str, value: np.ndarray) -> None:
        path = self.disk / f"{key}.npy"
        if path.exists():
            return
        tmp = path.with_name(f"{key}.tmp")
        try:
            with tmp.open("wb") as f:
                np.save(f, value, allow_pickle=False)
            tmp.replace(path)
            size = path.stat().st_size
        except OSError as e:  # a full disk must not fail the transcription
            logger.warning("Encoder cache: couldn't write %s: %s", path.name, e)
            tmp.unlink(missing_ok=True)
            return
        with self._lock:
            self._disk_bytes += size
            over = self._disk_bytes > self.disk_limit
        if over:
            self._trim_disk()

    def _trim_disk(self) -> None:
        """Delete the least recently used files down to 90% of the limit (not one file per new window)."""
        files = []
        for p in self.disk.glob("*.npy"):
            try:
                st = p.stat()
                files.append((st.st_mtime, st.st_size, p))
            except OSError:
                continue
        files.sort()
        total = sum(size for _, size, _ in files)
        for _, size, p in files:
            if total <= self.disk_limit * 0.9:
                break
            p.unlink(missing_ok=True)
            total -= size
        with self._lock:
            self._disk_bytes = total

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
        if self.disk is not None:
            for p in self.disk.glob("*.npy"):
                p.unlink(missing_ok=True)
            with self._lock:
                self._disk_bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled,
                    "memory": {"limit_mb": round(self.limit / 1e6, 1), "used_mb": round(self._bytes / 1e6, 1),
                               "windows": len(self._items)},
                    "disk": {"dir": str(self.disk), "limit_mb": round(self.disk_limit / 1e6, 1),
                             "used_mb": round(self._disk_bytes / 1e6, 1)} if self.disk is not None else None,
                    "hits": self.hits, "disk_hits": self.disk_hits, "misses": self.misses}


class SessionPool:
    """Open sessions by what they were opened from (a file digest) and how: asking again for a key whose session is
    still in use returns that session. Holds them weakly: a session no model uses any more is freed as before."""

    def __init__(self):
        self._live: dict[tuple, weakref.ref] = {}
        self._lock = threading.Lock()

    def open(self, key: tuple, create: Callable[[], object]) -> tuple[object, bool]:
        """(the session, whether it was already open)."""
        with self._lock:
            ref = self._live.get(key)
            found = ref() if ref is not None else None
        if found is not None:
            return found, True
        session = create()
        with self._lock:
            ref = self._live.get(key)
            found = ref() if ref is not None else None
            if found is not None:  # opened by another thread meanwhile: use that one
                return found, True
            try:
                self._live[key] = weakref.ref(session)
            except TypeError:  # can't be referenced weakly (a test stand-in): not shared
                pass
            for stale in [k for k, r in self._live.items() if r() is None]:
                del self._live[stale]
        return session, False

    def live(self) -> int:
        with self._lock:
            return sum(1 for r in self._live.values() if r() is not None)


ENCODERS = SessionPool()  # every model's encoder sessions, process-wide
