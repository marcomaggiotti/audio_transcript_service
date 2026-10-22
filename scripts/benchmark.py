"""Benchmark ONNX Whisper inference on this machine before promising latency numbers.

    python scripts/benchmark.py path/to/audio.mp3 --model-dir models/whisper-tiny --runs 3

Reports decode time, inference latency, real-time factor (inference / audio length;
below 1.0 is faster than real time) and peak resident memory.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import OnnxWhisperTranscriber  # noqa: E402
from app.audio import decode_audio, duration_seconds  # noqa: E402
from app.jobs import peak_memory_mb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--model-dir", default="models/whisper-tiny")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = OnnxWhisperTranscriber.from_dir(args.model_dir, threads=args.threads)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    audio = decode_audio(Path(args.audio).read_bytes())
    decode_s = time.perf_counter() - t0
    seconds = duration_seconds(audio)

    latencies = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = model.transcribe(audio, language=args.language)
        latencies.append(time.perf_counter() - t0)
        print(f"run {i + 1}: {latencies[-1]:.2f}s, {len(result.segments)} segments")

    median = statistics.median(latencies)
    print(f"\nmodel          {model.name} (loaded in {load_s:.2f}s)")
    print(f"audio          {seconds:.1f}s (decoded in {decode_s:.2f}s)")
    print(f"inference      median {median:.2f}s, min {min(latencies):.2f}s, max {max(latencies):.2f}s")
    print(f"real-time x    {median / seconds:.3f}")
    print(f"peak memory    {peak_memory_mb():.0f} MB")
    print(f"\n{result.text[:500]}")


if __name__ == "__main__":
    main()
