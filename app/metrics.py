"""Process memory: peak resident memory, which a benchmark can reset between runs (Linux)."""

import resource
import sys
from pathlib import Path

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def peak_memory_mb() -> float:
    """Peak resident memory: since the last reset_peak_memory() on Linux, else of the whole process."""
    try:
        for line in _STATUS.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def reset_peak_memory() -> bool:
    """Start measuring the peak again from the current memory use. False where the OS can't."""
    try:
        _CLEAR_REFS.write_text("5")
        return True
    except OSError:
        return False
