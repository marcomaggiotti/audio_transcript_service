"""Voice activity detection before Whisper, with Silero VAD (MIT licence, a 2 MB ONNX model).

Whisper invents text in silence and noise ("Thanks for watching", "Grazie per la visione") and its repetition loops
usually start there. With VAD on, Silero gives every 32 ms of audio a speech probability; the stretches of speech
(padded a little, joined across short pauses) are grouped into pieces of up to 30 s, cut at pauses, and only those
are transcribed, the silence between them left out. Whisper then hears only speech, no word is split at a fixed
30 s window edge, and silence costs no decoding time. The times of the segments are mapped back to the recording.

The speech detection follows silero-vad's get_speech_timestamps (v5.1.2), without torch.
"""

import logging
import zlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.asr import (
    COMPRESSION_RATIO_THRESHOLD,
    SAMPLE_RATE,
    DecodingOptions,
    Segment,
    Transcriber,
    TranscriptionResult,
    transcribe_intervals,
)

logger = logging.getLogger(__name__)

FRAME = 512  # samples per speech probability at 16 kHz: 32 ms
CONTEXT = 64  # samples of the previous frame the model sees with each one
MAX_PIECE_SECONDS = 29.0  # a piece and its padding fit in one 30 s window
SPEECH_PAD_MS = 200  # kept around each stretch of speech, so first and last sounds aren't clipped
MIN_SPEECH_MS = 250  # shorter blips (a click, a breath) are not speech
PREVIOUS_TEXT_CHARS = 600  # of the previous piece, prompted with the next one when previous text is on


class SileroVad:
    def __init__(self, session):
        self.session = session

    @classmethod
    def from_path(cls, path: str | Path) -> "SileroVad":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1  # one 32 ms frame at a time: 1 is fastest
        return cls(ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]))

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """The speech probability of every 32 ms frame (the last one padded with silence)."""
        n = -(-audio.shape[0] // FRAME)
        probs = np.empty(n, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(CONTEXT, dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        block = 30 * SAMPLE_RATE  # 16-bit input is converted 30 s at a time (half the RAM)
        for b in range(0, audio.shape[0], block):
            part = audio[b : b + block]
            if part.dtype == np.int16:
                part = part.astype(np.float32) / 32768.0
            part = np.pad(part, (0, -part.shape[0] % FRAME)).astype(np.float32, copy=False)
            for i in range(0, part.shape[0], FRAME):
                x = np.concatenate([context, part[i : i + FRAME]])
                out, state = self.session.run(None, {"input": x[None], "state": state, "sr": sr})
                probs[(b + i) // FRAME] = float(out[0, 0])
                context = x[-CONTEXT:]
        return probs

    def speech_spans(self, audio: np.ndarray, threshold: float = 0.5, min_silence_ms: int = 500,
                     speech_pad_ms: int = SPEECH_PAD_MS) -> list[tuple[float, float]]:
        return speech_spans(self.probabilities(audio), audio.shape[0], threshold, min_silence_ms, speech_pad_ms)


def speech_spans(probs: np.ndarray, n_samples: int, threshold: float = 0.5, min_silence_ms: int = 500,
                 speech_pad_ms: int = SPEECH_PAD_MS, min_speech_ms: int = MIN_SPEECH_MS,
                 max_speech_seconds: float = MAX_PIECE_SECONDS) -> list[tuple[float, float]]:
    """The stretches of speech, in seconds, from the frame probabilities (silero-vad's get_speech_timestamps).

    Speech starts at a frame above `threshold` and ends after `min_silence_ms` below `threshold - 0.15`; a stretch
    longer than `max_speech_seconds` is cut at its last pause of at least 98 ms (or hard, if it has none)."""
    sr = SAMPLE_RATE
    neg_threshold = max(threshold - 0.15, 0.01)
    min_speech = sr * min_speech_ms / 1000
    pad = int(sr * speech_pad_ms / 1000)
    max_speech = sr * max_speech_seconds - FRAME - 2 * pad
    min_silence = sr * min_silence_ms / 1000
    min_silence_at_max = sr * 98 / 1000

    speeches: list[list[int]] = []
    triggered, start = False, 0
    temp_end = prev_end = next_start = 0
    for i, p in enumerate(probs):
        at = FRAME * i
        if p >= threshold and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = at
        if p >= threshold and not triggered:
            triggered, start = True, at
            continue
        if triggered and at - start > max_speech:
            if prev_end:
                speeches.append([start, prev_end])
                if next_start < prev_end:  # still in the pause: not speech yet
                    triggered = False
                else:
                    start = next_start
                prev_end = next_start = temp_end = 0
            else:
                speeches.append([start, at])
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
        if p < neg_threshold and triggered:
            if not temp_end:
                temp_end = at
            if at - temp_end > min_silence_at_max:
                prev_end = temp_end
            if at - temp_end < min_silence:
                continue
            if temp_end - start > min_speech:
                speeches.append([start, temp_end])
            prev_end = next_start = temp_end = 0
            triggered = False
    if triggered and n_samples - start > min_speech:
        speeches.append([start, n_samples])

    for i, speech in enumerate(speeches):  # pad each side; a pause shorter than two pads is shared
        if i == 0:
            speech[0] = max(0, speech[0] - pad)
        if i != len(speeches) - 1:
            gap = speeches[i + 1][0] - speech[1]
            if gap < 2 * pad:
                speech[1] += gap // 2
                speeches[i + 1][0] = max(0, speeches[i + 1][0] - gap // 2)
            else:
                speech[1] = min(n_samples, speech[1] + pad)
                speeches[i + 1][0] = max(0, speeches[i + 1][0] - pad)
        else:
            speech[1] = min(n_samples, speech[1] + pad)
    return [(round(a / sr, 3), round(b / sr, 3)) for a, b in speeches if b > a]


def pieces(spans: list[tuple[float, float]], max_seconds: float = MAX_PIECE_SECONDS) -> list[list[tuple[float, float]]]:
    """The stretches of speech grouped, in order, into pieces of at most `max_seconds` of speech each."""
    out: list[list[tuple[float, float]]] = []
    length = 0.0
    for span in spans:
        seconds = span[1] - span[0]
        if out and length + seconds <= max_seconds:
            out[-1].append(span)
            length += seconds
        else:
            out.append([span])
            length = seconds
    return out


class _Timeline:
    """Maps a time in a piece (its speech joined, pauses left out) back to the recording."""

    def __init__(self, spans: list[tuple[float, float]]):
        self.spans = spans
        self.offsets = np.cumsum([0.0] + [b - a for a, b in spans])

    def __call__(self, t: float, end: bool = False) -> float:
        for (a, b), offset, next_offset in zip(self.spans, self.offsets, self.offsets[1:]):
            if t < next_offset or (end and t <= next_offset):
                return round(a + max(t - offset, 0.0), 2)
        return round(self.spans[-1][1], 2)


def _compression_ratio(text: str) -> float:
    data = text.encode()
    return len(data) / len(zlib.compress(data)) if data else 0.0


def transcribe_speech(transcriber: Transcriber, vad: SileroVad, audio: np.ndarray, language: str | None = None,
                      on_progress: Callable[[float], None] | None = None,
                      options: DecodingOptions | None = None) -> TranscriptionResult:
    """Transcribe only the speech: piece by piece, each up to 30 s of speech cut at pauses."""
    options = options or DecodingOptions()
    spans = vad.speech_spans(audio, options.vad_threshold, options.vad_min_silence_ms)
    detected = None if language in (None, "", "auto") else language
    segments, windows = [], []
    total = sum(b - a for a, b in spans) or 1.0
    done, previous = 0.0, ""
    for group in pieces(spans):
        piece = np.concatenate([audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] for a, b in group])
        prompt = " ".join(p for p in (options.initial_prompt, previous[-PREVIOUS_TEXT_CHARS:]) if p) or None
        result = transcriber.transcribe(piece, language=detected, options=replace(options, initial_prompt=prompt))
        detected = detected or result.language
        at = _Timeline(group)
        for s in result.segments:
            if s.text.strip():
                segments.append(Segment(at(s.start), at(s.end, end=True), s.text, s.avg_logprob, s.no_speech_prob,
                                        s.tokens))
        piece_windows = [replace(w, start=at(w.start)) for w in getattr(result, "windows", None) or []]
        # The previous piece's text prompts the next one, unless it loops (it would make the next loop too) or
        # was sampled at a high temperature (as openai/whisper).
        hot = any((w.attempts or [{}])[-1].get("temperature", 0) > 0.5 for w in piece_windows)
        looped = options.reset_on_loop and _compression_ratio(result.text) > COMPRESSION_RATIO_THRESHOLD
        if options.condition_on_previous_text and not hot and not looped:
            previous = (previous + " " + result.text).strip()
        else:
            previous = ""
            if looped and options.condition_on_previous_text and piece_windows:
                piece_windows[-1] = replace(piece_windows[-1], prompt_reset=True)
        windows += piece_windows
        done += sum(b - a for a, b in group)
        if on_progress:
            on_progress(min(1.0, done / total))
    if on_progress and not spans:
        on_progress(1.0)
    return TranscriptionResult(language=detected, segments=segments, windows=windows, speech=spans)


class SpeechOnly:
    """A transcriber that hears only the speech VAD finds (for transcribe_intervals and the benchmark)."""

    def __init__(self, transcriber: Transcriber, vad: SileroVad):
        self.transcriber, self.vad, self.name = transcriber, vad, transcriber.name

    def transcribe(self, audio, language=None, on_progress=None, options=None) -> TranscriptionResult:
        return transcribe_speech(self.transcriber, self.vad, audio, language, on_progress, options)


def transcribe_with(transcriber: Transcriber, audio: np.ndarray, language: str | None = None,
                    on_progress: Callable[[float], None] | None = None, options: DecodingOptions | None = None,
                    vad: SileroVad | None = None,
                    intervals: list[tuple[float, float]] | None = None) -> TranscriptionResult:
    """Transcribe with the options: only the speech when options.vad, on the given intervals when there are any."""
    options = options or DecodingOptions()
    if options.vad:
        if vad is None:
            raise RuntimeError("Voice activity detection is unavailable: the VAD model isn't installed "
                               "(VAD_MODEL_PATH)")
        transcriber = SpeechOnly(transcriber, vad)
    if intervals:
        return transcribe_intervals(transcriber, audio, intervals, language, on_progress, options)
    return transcriber.transcribe(audio, language=language, on_progress=on_progress, options=options)


def load_vad(path: str | None) -> tuple[SileroVad | None, str | None]:
    """The VAD model, or None and why not."""
    if not path or not Path(path).exists():
        return None, f"no VAD model at {path}"
    try:
        return SileroVad.from_path(path), None
    except Exception as e:  # a broken file: transcription still works without VAD
        logger.exception("VAD model didn't load")
        return None, str(e) or e.__class__.__name__
