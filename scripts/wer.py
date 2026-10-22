"""Word error rate of Whisper models and decoding options on reference clips.

    python scripts/wer.py data/benchmark/clips --model-dir models/whisper-tiny --model-dir models/whisper-base \
        --variant greedy --variant fallback_previous --language it

CLIPS is a folder of audio files, each with a same-name .txt holding its correct transcript
(clip01.mp3 + clip01.txt). The Benchmark tab saves its reference clips in exactly this form, in
BENCHMARK_DIR/clips. Prints WER, error counts, real-time factor and peak memory per model and variant,
and with --details the worst clips with their transcripts.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import OnnxWhisperTranscriber  # noqa: E402
from app.benchmark import VARIANTS, clips_from_dir, evaluate, variant_options  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, help="Folder of audio files with same-name .txt references")
    parser.add_argument("--model-dir", action="append", required=True, help="Repeat to compare models")
    parser.add_argument("--variant", action="append", choices=list(VARIANTS),
                        help="Decoding variant; repeat to compare (default: all)")
    parser.add_argument("--language", default="auto", help="e.g. en, it, de; auto detects per clip")
    parser.add_argument("--initial-prompt", help="Vocabulary or style hint given to every window")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--details", action="store_true", help="Show the worst clips of every run")
    args = parser.parse_args()

    clips = clips_from_dir(args.clips)
    if not clips:
        print(f"No audio files with a same-name .txt reference in {args.clips}", file=sys.stderr)
        return 1
    print(f"{len(clips)} clips from {args.clips}\n")
    language = None if args.language == "auto" else args.language
    references = {c.id: c.reference for c in clips}

    rows = []
    for model_dir in args.model_dir:
        t0 = time.perf_counter()
        model = OnnxWhisperTranscriber.from_dir(model_dir, threads=args.threads)
        load_s = time.perf_counter() - t0
        for variant in args.variant or list(VARIANTS):
            print(f"{model.name} · {VARIANTS[variant].label} …", flush=True)
            result = evaluate(model, clips, language, variant_options(variant, args.initial_prompt))
            rows.append((model.name, VARIANTS[variant].label, result, load_s))
            if args.details:
                for clip in sorted(result.clips, key=lambda c: c.wer, reverse=True)[:5]:
                    print(f"  {clip.wer:6.1%}  {clip.name}")
                    print(f"          ref: {references[clip.clip_id]}")
                    print(f"          hyp: {clip.hypothesis}")
        del model
        gc.collect()

    print(f"\n{'model':<28} {'decoding':<26} {'WER':>7} {'sub':>5} {'del':>5} {'ins':>5} {'RTF':>6} "
          f"{'peak MB':>8} {'load s':>7}")
    best = min((r[2].wer for r in rows if r[2].wer is not None), default=None)
    for name, label, result, load_s in rows:
        marker = " *" if result.wer == best else ""
        wer = f"{result.wer:.1%}" if result.wer is not None else "–"
        print(f"{name:<28} {label:<26} {wer:>7} {result.substitutions:>5} {result.deletions:>5} "
              f"{result.insertions:>5} {result.real_time_factor or 0:>6.3f} {result.peak_memory_mb:>8.0f} "
              f"{load_s:>7.2f}{marker}")
    print(f"\nWER over {sum(c.reference_words for c in rows[0][2].clips)} reference words. * = lowest WER. "
          "RTF below 1 is faster than real time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
