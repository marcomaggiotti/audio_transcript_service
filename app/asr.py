"""Speech recognition with a Whisper model exported to ONNX, run through ONNX Runtime.

Expects the directory layout produced by scripts/export_whisper_onnx.py (optimum export, then INT8):

    encoder_model.onnx          input_features -> last_hidden_state
    decoder_model_merged.onnx   input_ids + encoder_hidden_states + past_key_values.* + use_cache_branch
                                -> logits + present.* (KV cache: one new token per decoder call)
      or decoder_model.onnx     input_ids + encoder_hidden_states -> logits (legacy, no KV cache:
                                every step re-runs the whole prefix and returns [1, n, vocab] logits)
    tokenizer.json
    generation_config.json    (optional, provides the suppressed token ids)
    preprocessor_config.json  (optional, provides n_mels)

English only: the service runs English-only Whisper models (openai/whisper-*.en, by default whisper-tiny.en),
whose prompt is just <|startoftranscript|>. A multilingual export is refused when it loads.

Decoding is greedy with Whisper's timestamp rules, so every transcript comes back as
segments with start/end times. Long audio is processed in 30 s windows that advance
to the last complete timestamp, the same "seek" strategy openai/whisper uses.
DecodingOptions adds openai/whisper's temperature fallback (re-decode a window whose text
repeats or is unsure), prompting with the previous text, and an initial prompt.
"""

import json
import logging
import os
import time
import zlib
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from app.encoder_cache import ENCODERS, EncoderCache, file_digest, mel_digest

from app.features import (
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

logger = logging.getLogger(__name__)



class Cancelled(RuntimeError):
    """Raised inside a transcription whose job was cancelled (DELETE /jobs/{id})."""


# Set by the job worker around a transcription, in its own thread: called before every decoder step, it raises
# Cancelled once the job is cancelled, so a stopped job frees the CPU within a step instead of running to the end.
cancel_check: ContextVar[Callable[[], None] | None] = ContextVar("cancel_check", default=None)


def check_cancelled() -> None:
    if (check := cancel_check.get()) is not None:
        check()


TIME_PRECISION = 0.02  # seconds per timestamp token
MAX_INITIAL_TIMESTAMP = 1.0  # first segment must start within 1 s of the window
N_TEXT_CTX = 448  # decoder positions: prompt + sampled tokens must fit
MAX_PROMPT_TOKENS = N_TEXT_CTX // 2 - 1  # previous-text tokens kept in the prompt (as openai/whisper)

# Temperature fallback, with openai/whisper's thresholds: a window is decoded again, sampling at the
# next temperature, when its text compresses too well (repetition loop) or its tokens are unsure.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
# With fallback on, a window Whisper thinks is silent (and whose text is unsure) produces no segments.
NO_SPEECH_THRESHOLD = 0.6


@dataclass(frozen=True)
class DecodingOptions:
    """How each 30 s window is decoded. The defaults are plain greedy decoding."""

    temperature_fallback: bool = False  # re-decode a window at rising temperatures when its text looks wrong
    condition_on_previous_text: bool = False  # prompt each window with the text decoded before it
    initial_prompt: str | None = None  # vocabulary or style hint, prompted on every window
    seed: int = 0  # for the sampled (temperature > 0) attempts, so runs are repeatable
    beam_size: int = 1  # 1 = greedy; more keeps that many candidate texts at temperature 0 (openai/whisper: 5)
    reset_on_loop: bool = True  # don't prompt the next window with a window whose text repeats itself
    # Keep the decoder's keys/values of the tokens already written, so each step computes only the new one.
    # Off: every step recomputes them from the whole text so far, and the cross-attention keys/values of all
    # 1500 encoder frames too: the same tokens, about 10x the decoder work (times the beams with beam search).
    # It is a speed-up only: the tokens Whisper conditions on are the same either way. Exports without a merged
    # decoder (decoder_model.onnx only) never have it.
    kv_cache: bool = True
    # Voice activity detection (app/vad.py): Whisper hears only the speech, cut at pauses into pieces up to 30 s.
    vad: bool = False
    vad_threshold: float = 0.5  # speech probability above which a 32 ms frame is speech
    vad_min_silence_ms: int = 500  # a pause at least this long ends a stretch of speech


@dataclass
class Segment:
    start: float
    end: float
    text: str
    # How sure Whisper was (None from transcribers that don't measure it):
    avg_logprob: float | None = None  # mean log-probability of the segment's text tokens
    no_speech_prob: float | None = None  # Whisper's estimate that its 30 s window is silence
    tokens: list[int] = field(default_factory=list, repr=False, compare=False)  # text token ids


@dataclass
class WindowTrace:
    """What happened in one 30 s window: where it started, what it kept, how often it was decoded."""

    start: float  # seconds: the seek position
    length: float  # seconds of audio in the window (30, or less at the end; padded with silence to 30)
    consumed: float  # seconds the seek moved on: to the last complete timestamp (at least 1 s)
    no_speech_prob: float | None
    skipped: bool  # dropped as silence
    segments: int
    attempts: list[dict] = field(default_factory=list)  # each decode: temperature, compression_ratio, avg_logprob
    prompt_reset: bool = False  # its text looped, so the next window wasn't prompted with it
    encoder_cached: bool = False  # the encoder's output came from the encoder cache (app/encoder_cache.py)


@dataclass
class TranscriptionResult:
    segments: list[Segment] = field(default_factory=list)
    windows: list[WindowTrace] = field(default_factory=list)
    speech: list[tuple[float, float]] | None = None  # with VAD: the stretches of speech Whisper heard

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


@dataclass
class WindowResult:
    segments: list[Segment]  # relative to the window
    consumed: float  # seconds of the window covered by the segments
    avg_logprob: float
    compression_ratio: float
    temperature: float
    attempts: list[dict] = field(default_factory=list)  # every decode of this window, the last one kept


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, on_progress: Callable[[float], None] | None = None,
                   options: DecodingOptions | None = None) -> TranscriptionResult: ...


MAX_INTERVALS = 5000
MAX_BEAM_SIZE = 8


def transcribe_intervals(transcriber: Transcriber, audio: np.ndarray, intervals: list[tuple[float, float]],
                         on_progress: Callable[[float], None] | None = None,
                         options: DecodingOptions | None = None) -> TranscriptionResult:
    """Transcribe each [start, end) of the audio on its own and return exactly one segment per interval, with the
    interval's own times (its text empty when Whisper heard nothing). Nothing is carried from one interval to the
    next (no previous-text prompt), so text can't leak across them: for transcribing a copy of a recording on the
    original's lines."""
    options = replace(options or DecodingOptions(), condition_on_previous_text=False)
    duration = audio.shape[0] / SAMPLE_RATE
    segments: list[Segment] = []
    windows: list[WindowTrace] = []
    for k, (start, end) in enumerate(intervals):
        start, end = max(0.0, float(start)), min(float(end), duration)
        piece = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        if end - start >= 0.1 and piece.shape[0]:
            result = transcriber.transcribe(piece, options=options)
            texts = [s.text.strip() for s in result.segments if s.text.strip()]
            logprobs = [s.avg_logprob for s in result.segments if s.avg_logprob is not None]
            no_speech = [s.no_speech_prob for s in result.segments if s.no_speech_prob is not None]
            segments.append(Segment(round(start, 2), round(max(end, start), 2), " ".join(texts),
                                    round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
                                    max(no_speech) if no_speech else None,
                                    [t for s in result.segments for t in s.tokens]))
            windows += [replace(w, start=round(start + w.start, 2)) for w in getattr(result, "windows", None) or []]
        else:  # outside the audio, or too short to decode
            segments.append(Segment(round(min(start, duration), 2), round(max(end, min(start, duration)), 2), ""))
        if on_progress:
            on_progress((k + 1) / len(intervals))
    return TranscriptionResult(segments=segments, windows=windows)


class _PrefixSession:
    """Decoder without a KV cache: every step re-runs the whole prefix. Used for legacy exports
    (decoder_model.onnx), and for the merged decoder when kv_cache is off: `feeds` then select its first-call
    branch (use_cache_branch false, empty past), which takes any number of tokens."""

    def __init__(self, decoder, encoded: np.ndarray, feeds: dict | None = None):
        self.decoder, self.encoded, self.feeds = decoder, encoded, feeds or {}

    def __call__(self, tokens: list[int]) -> np.ndarray:
        check_cancelled()
        logits = self.decoder.run(["logits"], {"input_ids": np.array([tokens], dtype=np.int64),
                                               "encoder_hidden_states": self.encoded, **self.feeds})[0]
        return logits[0, -1].astype(np.float32)

    def fork(self) -> "_PrefixSession":
        return self


class _CachedSession:
    """Merged decoder with a KV cache: each call feeds only the tokens added since the previous one and keeps the
    self-attention K/V in between; the cross-attention K/V (from encoded) are computed on the first call. fork()
    copies the state for a beam that continues from here: the arrays are replaced on every call, never changed in
    place, so sharing them is safe."""

    def __init__(self, decoder, outputs: list[str], encoded: np.ndarray, past: dict, fed: int):
        self.decoder, self.outputs, self.encoded, self.past, self.fed = decoder, outputs, encoded, past, fed

    def __call__(self, tokens: list[int]) -> np.ndarray:
        check_cancelled()
        first = self.fed == 0
        outputs = self.decoder.run(None, {
            "input_ids": np.array([tokens[self.fed:]], dtype=np.int64),
            "encoder_hidden_states": self.encoded,
            "use_cache_branch": np.array([not first]),
            **self.past,
        })
        logits = None
        past = dict(self.past)
        for name, value in zip(self.outputs, outputs):
            if name == "logits":
                logits = value
            elif first or ".decoder." in name:  # later calls return placeholder cross-attention K/V
                past["past_key_values" + name[len("present"):]] = value
        self.past, self.fed = past, len(tokens)
        return logits[0, -1].astype(np.float32)

    def fork(self) -> "_CachedSession":
        return _CachedSession(self.decoder, self.outputs, self.encoded, self.past, self.fed)


class ModelLoadError(RuntimeError):
    pass


PRECISIONS = ("fp32", "int8", "int4")
# ONNX Runtime rewrites the graph when it opens a model (GraphOptimizationLevel): basic = constant folding and redundant
# nodes removed; extended = fused operators (attention, LayerNorm, GELU, MatMul+Add); layout = also the CPU's memory
# layout; all = everything. Higher levels take longer to open and usually run faster.
GRAPH_LEVELS = ("disabled", "basic", "extended", "layout", "all")


def graph_level(name: str):
    import onnxruntime as ort

    levels = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
    if name == "layout":  # ONNX Runtime 1.22+; before, layout came only with all
        return getattr(ort.GraphOptimizationLevel, "ORT_ENABLE_LAYOUT", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    return levels[name]


def _release_memory() -> None:
    """Hand freed memory back to the OS (glibc keeps it otherwise, so RSS would stay at the swap's peak)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc (macOS, musl): nothing to do
        pass


def onnx_precision(path: Path) -> str:
    """'int4' if its weights were quantized to 4 bits (MatMulNBits), 'int8' if the graph was dynamically quantized
    (scripts/export_whisper_onnx.py), else 'fp32'.

    Op types are stored as plain strings in the protobuf, so this needs no onnx package and no full parse.
    """
    import mmap

    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data.find(b"MatMulNBits") >= 0:
            return "int4"
        return "int8" if data.find(b"DynamicQuantizeLinear") >= 0 or data.find(b"MatMulInteger") >= 0 else "fp32"


@dataclass
class WhisperTokens:
    """Special token ids, resolved from the tokenizer and generation config."""

    eot: int
    sot: int
    no_timestamps: int
    timestamp_begin: int
    suppress: list[int]
    no_speech: int | None = None  # <|nocaptions|> in whisper-*.en (<|nospeech|> in newer checkpoints)
    prev: int | None = None  # <|startofprev|>: starts the previous-text prompt


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def is_multilingual_config(generation_config: dict) -> bool:
    """whisper-*.en say is_multilingual: false; a multilingual config says true, or lists its languages."""
    return bool(generation_config.get("is_multilingual", bool(generation_config.get("lang_to_id"))))


def _resolve_tokens(tokenizer, generation_config: dict) -> WhisperTokens:
    def tid(token: str) -> int | None:
        return tokenizer.token_to_id(token)

    eot, sot = tid("<|endoftext|>"), tid("<|startoftranscript|>")
    no_ts = generation_config.get("no_timestamps_token_id") or tid("<|notimestamps|>")
    if eot is None or sot is None or no_ts is None:
        raise ModelLoadError("tokenizer.json is missing Whisper special tokens")
    if is_multilingual_config(generation_config):
        raise ModelLoadError("This is a multilingual Whisper model; the service runs English-only ones: export "
                             "openai/whisper-tiny.en (or another openai/whisper-*.en)")

    return WhisperTokens(
        eot=eot,
        sot=sot,
        no_timestamps=no_ts,
        timestamp_begin=no_ts + 1,
        suppress=[t for t in generation_config.get("suppress_tokens", []) if t >= 0],
        no_speech=tid("<|nospeech|>") if tid("<|nospeech|>") is not None else tid("<|nocaptions|>"),
        prev=tid("<|startofprev|>"),
    )


class OnnxWhisperTranscriber:
    """Greedy Whisper decoding on ONNX Runtime (CPU by default)."""

    def __init__(self, encoder, decoder, tokenizer, tokens: WhisperTokens, n_mels: int = 80,
                 max_new_tokens: int = 224, name: str = "whisper-onnx"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.n_mels = n_mels
        self.max_new_tokens = max_new_tokens
        self.name = name
        # Precision of each part, and the files of every precision exported (from_dir fills these in).
        self.precision: dict[str, str] | None = None
        self.variants: dict[str, dict[str, Path]] = {}
        self._load_session: Callable[[Path], object] | None = None
        self._open_session: Callable[[Path, str], object] | None = None  # (file, graph optimization level)
        self.graph_optimization: str | None = None  # the level the loaded sessions were opened with
        self.session_seconds: float | None = None  # how long opening them took (optimizing the graph included)
        # Encoder outputs kept and reused (shared by every model of the service), and the file and graph optimization
        # level the encoder session in use was opened from: its key there. None: not cached (test stand-ins).
        self.encoder_cache: EncoderCache | None = None
        self._encoder_source: tuple[Path, str] | None = None
        self._open_encoder: Callable[[Path, str], object] | None = None
        self._bind(encoder, decoder)

    def _bind(self, encoder, decoder) -> None:
        """Use these sessions, and read the input/output layout the decoding loop needs from them."""
        self.encoder, self.decoder = encoder, decoder
        self._encoder_input = encoder.get_inputs()[0].name
        decoder_inputs = {i.name: i for i in decoder.get_inputs()}
        if {"input_ids", "encoder_hidden_states"} - decoder_inputs.keys():
            raise ModelLoadError(
                f"Unsupported decoder inputs {sorted(decoder_inputs)}; "
                "export with scripts/export_whisper_onnx.py (decoder_model_merged.onnx or decoder_model.onnx)"
            )
        # optimum's merged decoder: an If node picks the first-call branch (no past) or the cached one.
        self._kv_cache = "use_cache_branch" in decoder_inputs
        self._decoder_outputs = [o.name for o in decoder.get_outputs()] if self._kv_cache else ["logits"]
        # past_key_values.<i>.{decoder,encoder}.{key,value}: [batch, heads, seq, head_dim], seq = 0 at start
        self._empty_past = {name: (1, inp.shape[1], 0, inp.shape[3]) for name, inp in decoder_inputs.items()
                            if name.startswith("past_key_values.")}

    @classmethod
    def from_dir(cls, model_dir: str, max_new_tokens: int = 224, threads: int = 0,
                 low_memory: bool = False, graph_optimization: str = "all",
                 encoder_cache: EncoderCache | None = None) -> "OnnxWhisperTranscriber":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(model_dir)
        decoder_file = next((n for n in ("decoder_model_merged.onnx", "decoder_model.onnx") if (path / n).exists()),
                            "decoder_model_merged.onnx")
        for name in ("encoder_model.onnx", decoder_file, "tokenizer.json"):
            if not (path / name).exists():
                raise ModelLoadError(
                    f"{path / name} not found. Export a model first: python scripts/export_whisper_onnx.py"
                )

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        # No arena: it keeps the encoder's peak (attention, mel) allocated after every window. With the KV
        # cache each decoder call is tiny, so plain malloc costs nothing measurable.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        if low_memory:  # ~-25 MB RSS (no pre-packed weight copies), decoder steps ~2x slower
            options.add_session_config_entry("session.disable_prepacking", "1")
        providers = ["CPUExecutionProvider"]

        def open_session(file: Path, level: str):
            options.graph_optimization_level = graph_level(level)  # read when the session is created
            return ort.InferenceSession(str(file), options, providers=providers)

        def open_encoder(file: Path, level: str):
            """The encoder's session: the one already open when a loaded model has this very encoder file (a fine-tuned
            model and its base, the encoder frozen), else a new one."""
            key = ("encoder", file_digest(file), level, threads, low_memory)
            return ENCODERS.open(key, lambda: open_session(file, level))[0]

        if graph_optimization not in GRAPH_LEVELS:
            logger.warning("Unknown graph optimization level %r (%s): using all", graph_optimization, ", ".join(GRAPH_LEVELS))
            graph_optimization = "all"
        t0 = time.perf_counter()
        encoder = open_encoder(path / "encoder_model.onnx", graph_optimization)
        decoder = open_session(path / decoder_file, graph_optimization)
        opened = time.perf_counter() - t0
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        tokens = _resolve_tokens(tokenizer, _load_json(path / "generation_config.json"))
        n_mels = _load_json(path / "preprocessor_config.json").get("feature_size", 80)
        model_id = path / "model_id.txt"  # written by scripts/export_whisper_onnx.py
        name = model_id.read_text().strip() if model_id.exists() else os.path.basename(path.resolve())
        transcriber = cls(encoder, decoder, tokenizer, tokens, n_mels=n_mels, max_new_tokens=max_new_tokens, name=name)
        transcriber.precision = {"encoder": onnx_precision(path / "encoder_model.onnx"),
                                 "decoder": onnx_precision(path / decoder_file)}
        # The other precision of each part, if the export kept it (encoder_model.int8.onnx, ...).
        for part, default_file in (("encoder", "encoder_model.onnx"), ("decoder", decoder_file)):
            stem = default_file.removesuffix(".onnx")
            files = {transcriber.precision[part]: path / default_file}
            for p in PRECISIONS:
                if p not in files and (path / f"{stem}.{p}.onnx").exists():
                    files[p] = path / f"{stem}.{p}.onnx"
            transcriber.variants[part] = files
        transcriber._open_session = open_session
        transcriber._open_encoder = open_encoder
        transcriber._encoder_source = (path / "encoder_model.onnx", graph_optimization)
        transcriber.encoder_cache = encoder_cache
        transcriber._load_session = lambda file: open_session(file, transcriber.graph_optimization or graph_optimization)
        transcriber.graph_optimization, transcriber.session_seconds = graph_optimization, round(opened, 3)
        return transcriber

    def weights_mb(self, used: dict[str, str] | None = None) -> float | None:
        """The size of the weights a job ran with: the encoder and decoder files of the precisions used (with their
        external data, if any). None when the files aren't known."""
        used = used or self.precision or {}
        try:
            files = [self.variants[part][used[part]] for part in ("encoder", "decoder")]
        except KeyError:
            return None
        sizes = [f.stat().st_size + (f.with_name(f.name + "_data").stat().st_size if f.with_name(f.name + "_data").exists() else 0)
                 for f in files if f.exists()]
        return round(sum(sizes) / 1e6, 1) if len(sizes) == 2 else None

    @property
    def available_precisions(self) -> dict[str, list[str]]:
        """Precisions a job can ask for, per part ({} if unknown: then only the default)."""
        return {part: sorted(files) for part, files in self.variants.items()}

    @contextmanager
    def precision_for(self, encoder: str | None = None, decoder: str | None = None):
        """Transcribe with these precisions inside the block (None = the default), then go back.

        A part in another precision is swapped in: the default one is released first, the requested one
        loaded (1-2 s for tiny), and after the block the default is loaded again. So two versions of a part
        are never in memory together (the all-fp32 pair alone peaks ~320 MB on tiny). Jobs run one at a
        time (WORKER_CONCURRENCY=1), so swapping is safe. Yields the precision actually used per part.
        """
        default = dict(self.precision or {})
        wanted = {"encoder": encoder or default.get("encoder"), "decoder": decoder or default.get("decoder")}
        for part, p in wanted.items():
            if p != default.get(part) and p not in self.variants.get(part, {}):
                raise ValueError(f"{part} precision {p} isn't available "
                                 f"(this export has {', '.join(self.variants.get(part) or [default.get(part) or '?'])})")
        swapped = [part for part in ("encoder", "decoder") if wanted[part] != default.get(part)]
        try:
            self._swap({part: wanted[part] for part in swapped})
            yield wanted
        finally:
            if swapped:
                self._swap({part: default[part] for part in swapped})
                _release_memory()  # the variant's weights are freed by the swap: return them too

    @contextmanager
    def optimization_for(self, level: str | None = None):
        """Transcribe with sessions opened at this graph optimization level inside the block (None = as loaded), then
        go back. Another level reopens both parts (the old sessions released first, as a precision swap does): the
        graph is optimized when a session opens. Yields {level, session_seconds}: the level used and how long opening
        the sessions at it took (this job's reopening, or the service's start)."""
        default = self.graph_optimization
        if not level or level == default or self._open_session is None:
            yield {"level": default, "session_seconds": self.session_seconds}
            return
        if level not in GRAPH_LEVELS:
            raise ValueError(f"Unknown graph optimization level {level} ({', '.join(GRAPH_LEVELS)})")
        files = {part: self.variants[part][self.precision[part]] for part in ("encoder", "decoder")}

        def reopen(at: str) -> float:
            self.encoder = self.decoder = None  # release before reopening: never two copies in memory
            _release_memory()
            t0 = time.perf_counter()
            encoder = (self._open_encoder or self._open_session)(files["encoder"], at)
            decoder = self._open_session(files["decoder"], at)
            if self._encoder_source is not None:
                self._encoder_source = (files["encoder"], at)
            self.graph_optimization = at
            self._bind(encoder, decoder)
            return round(time.perf_counter() - t0, 3)

        try:
            yield {"level": level, "session_seconds": reopen(level)}
        finally:
            reopen(default)
            _release_memory()

    def _swap(self, precisions: dict[str, str]) -> None:
        """Replace these parts' sessions with the given precision, releasing the old session first."""
        if not precisions:
            return
        sessions = {"encoder": self.encoder, "decoder": self.decoder}
        for part in precisions:
            sessions[part] = None
            setattr(self, part, None)  # drop the last reference before loading the replacement
        _release_memory()
        for part, p in precisions.items():
            file = self.variants[part][p]
            if part == "encoder" and self._open_encoder is not None:
                level = self.graph_optimization or "all"
                sessions[part] = self._open_encoder(file, level)
                self._encoder_source = (file, level)
            else:
                sessions[part] = self._load_session(file)
        self._bind(sessions["encoder"], sessions["decoder"])

    @property
    def encoder_digest(self) -> str | None:
        """A digest of the encoder file in use: the same for models that share their encoder."""
        return file_digest(self._encoder_source[0]) if self._encoder_source else None

    def _encode(self, mel: np.ndarray) -> tuple[np.ndarray, bool]:
        """The encoder's output for one window, and whether it came from the encoder cache: a window this encoder
        (the same file, at the same graph optimization level) already encoded isn't encoded again."""
        cache, source = self.encoder_cache, self._encoder_source
        if cache is None or source is None or not cache.enabled:
            return self.encoder.run(None, {self._encoder_input: mel[None]})[0], False
        key = f"{file_digest(source[0])}-{source[1]}-{mel_digest(mel)}"
        found = cache.get(key)
        if found is not None:
            return found, True
        return cache.put(key, self.encoder.run(None, {self._encoder_input: mel[None]})[0]), False


    # ------------------------------------------------------------------ public

    def transcribe(self, audio: np.ndarray, on_progress: Callable[[float], None] | None = None,
                   options: DecodingOptions | None = None) -> TranscriptionResult:
        """`on_progress` gets the share of the audio transcribed (0-1) after each 30 s window."""
        options = options or DecodingOptions()
        rng = np.random.default_rng(options.seed)
        initial = self._encode_prompt(options.initial_prompt)
        previous: list[int] = []  # text tokens of the windows decoded so far, for the next prompt
        duration = audio.shape[0] / SAMPLE_RATE
        segments: list[Segment] = []
        windows: list[WindowTrace] = []
        seek = 0  # in samples

        while seek < audio.shape[0]:
            window = audio[seek : seek + N_SAMPLES]
            if window.dtype == np.int16:  # 16-bit input: convert one window at a time (half the RAM)
                window = window.astype(np.float32) / 32768.0
            window_seconds = window.shape[0] / SAMPLE_RATE
            if window_seconds < 0.1:  # trailing sliver, nothing useful to decode
                break
            offset = seek / SAMPLE_RATE

            mel = log_mel_spectrogram(pad_or_trim(window), self.n_mels)[:, :N_FRAMES]
            encoded, cached = self._encode(mel)

            # One decoder step after <|startoftranscript|> gives the no-speech odds.
            no_speech = self._no_speech_prob(self._next_logits([self.tokens.sot], encoded))

            result = self._decode_with_fallback(encoded, window_seconds, initial + previous, no_speech, options, rng)
            silent = (options.temperature_fallback and no_speech is not None and no_speech > NO_SPEECH_THRESHOLD
                      and result.avg_logprob < LOGPROB_THRESHOLD)
            kept = len(segments)
            reset = False
            if silent:  # Whisper's no-speech skip: drop the window's text, move past the whole window
                result.consumed = window_seconds
                previous = []
            else:
                for seg in result.segments:
                    start = min(offset + seg.start, duration)
                    end = min(offset + seg.end, duration)
                    if seg.text and end > start:
                        segments.append(Segment(round(start, 2), round(end, 2), seg.text,
                                                seg.avg_logprob, no_speech, seg.tokens))
                # Text sampled at a high temperature is a poor guide for the next window (as openai/whisper), and
                # text that loops would make the next window loop too: start it afresh.
                looped = options.reset_on_loop and result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                if options.condition_on_previous_text and result.temperature <= 0.5 and not looped:
                    previous = (previous + [t for seg in result.segments for t in seg.tokens])[-MAX_PROMPT_TOKENS:]
                else:
                    previous = []
                reset = options.condition_on_previous_text and looped
            moved = max(int(result.consumed * SAMPLE_RATE), SAMPLE_RATE)  # always move forward
            windows.append(WindowTrace(round(offset, 2), round(window_seconds, 2), round(moved / SAMPLE_RATE, 2),
                                       None if no_speech is None else round(no_speech, 4), silent,
                                       len(segments) - kept, result.attempts, reset, cached))
            seek += moved
            if on_progress:
                on_progress(min(1.0, seek / audio.shape[0]))

        return TranscriptionResult(segments=segments, windows=windows)

    def _no_speech_prob(self, logits: np.ndarray) -> float | None:
        if self.tokens.no_speech is None:
            return None
        log_probs = logits - np.logaddexp.reduce(logits)
        return round(float(np.exp(log_probs[self.tokens.no_speech])), 4)

    # ---------------------------------------------------------------- decoding

    def _encode_prompt(self, text: str | None) -> list[int]:
        if not text or not text.strip() or self.tokens.prev is None:
            return []
        ids = self.tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids
        return [t for t in ids if t < self.tokens.eot][-MAX_PROMPT_TOKENS:]

    def _prompt(self, previous: list[int] | None = None) -> list[int]:
        """<|startoftranscript|>, after the previous text (<|startofprev|> ...) when there is some."""
        if previous and self.tokens.prev is not None:
            return [self.tokens.prev, *previous[-MAX_PROMPT_TOKENS:], self.tokens.sot]
        return [self.tokens.sot]

    def _next_logits(self, tokens: list[int], encoded: np.ndarray) -> np.ndarray:
        return self._decoder_session(encoded)(tokens)

    @property
    def kv_cache_available(self) -> bool:
        """Whether this export's decoder has a KV cache (the merged decoder); legacy exports re-run the prefix."""
        return self._kv_cache

    def _decoder_session(self, encoded: np.ndarray, kv_cache: bool = True):
        """Return step(tokens) -> logits for the next token, for one window.

        With the merged decoder and kv_cache, step() feeds only the tokens added since the previous call and keeps
        the self-attention K/V in between; the cross-attention K/V (from encoded) are computed on the first call.
        Without, every step runs the decoder over all the tokens so far.
        """
        past = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._empty_past.items()}
        if not self._kv_cache:
            return _PrefixSession(self.decoder, encoded)
        if not kv_cache:
            return _PrefixSession(self.decoder, encoded, {"use_cache_branch": np.array([False]), **past})
        return _CachedSession(self.decoder, self._decoder_outputs, encoded, past, 0)

    def _decode_with_fallback(self, encoded: np.ndarray, window_seconds: float,
                              previous: list[int], no_speech: float | None, options: DecodingOptions,
                              rng: np.random.Generator) -> WindowResult:
        temperatures = FALLBACK_TEMPERATURES if options.temperature_fallback else (0.0,)
        tried: list[dict] = []
        for temperature in temperatures:
            result = self._decode_window(encoded, window_seconds, previous, temperature, rng,
                                         beam_size=options.beam_size, kv_cache=options.kv_cache)
            tried.append({"temperature": temperature, "compression_ratio": round(result.compression_ratio, 3),
                          "avg_logprob": round(result.avg_logprob, 4),
                          "beam_size": options.beam_size if temperature == 0 and options.beam_size > 1 else 1})
            result.attempts = tried
            looks_wrong = (result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                           or result.avg_logprob < LOGPROB_THRESHOLD)
            if not looks_wrong or (no_speech is not None and no_speech > NO_SPEECH_THRESHOLD):
                break  # good enough, or silence (skipped by the caller, retrying won't help)
        return result

    def _decode_window(self, encoded: np.ndarray, window_seconds: float,
                       previous: list[int] | None = None, temperature: float = 0.0,
                       rng: np.random.Generator | None = None, beam_size: int = 1,
                       kv_cache: bool = True) -> WindowResult:
        """Decode one 30 s window: greedy (or beam search) at temperature 0, otherwise sampled."""
        prompt = self._prompt(previous)
        steps = min(self.max_new_tokens, N_TEXT_CTX - len(prompt))
        if temperature == 0 and beam_size > 1:
            sampled, logprobs = self._beam_search(encoded, prompt, steps, beam_size, kv_cache)
        else:
            sampled, logprobs = self._sample(encoded, prompt, steps, temperature, rng, kv_cache)
        segments, consumed = self._split_segments(sampled, window_seconds, logprobs)
        text = self._decode_text([t for t in sampled if t < self.tokens.eot]).encode()
        return WindowResult(
            segments=segments,
            consumed=consumed,
            avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
            compression_ratio=len(text) / len(zlib.compress(text)) if text else 0.0,
            temperature=temperature,
        )

    def _sample(self, encoded: np.ndarray, prompt: list[int], steps: int, temperature: float,
                rng: np.random.Generator | None, kv_cache: bool = True) -> tuple[list[int], list[float]]:
        """One token at a time: the most likely (temperature 0) or a sampled one. Returns the tokens after the
        prompt and the log-probability of each, after the filters (as Whisper scores them)."""
        tokens = list(prompt)
        logprobs: list[float] = []
        next_logits = self._decoder_session(encoded, kv_cache)
        for _ in range(steps):
            logits = next_logits(tokens)
            self._apply_rules(logits, tokens[len(prompt):])
            finite = np.isfinite(logits)
            if temperature > 0:
                scaled = np.where(finite, logits / temperature, -np.inf)
                probs = np.exp(scaled - scaled[finite].max())
                next_token = int((rng or np.random.default_rng()).choice(probs.shape[0], p=probs / probs.sum()))
            else:
                next_token = int(np.argmax(logits))
            logprobs.append(float(logits[next_token] - np.logaddexp.reduce(logits[finite])))
            tokens.append(next_token)
            if next_token == self.tokens.eot:
                break
        return tokens[len(prompt):], logprobs

    def _beam_search(self, encoded: np.ndarray, prompt: list[int], steps: int,
                     beam_size: int, kv_cache: bool = True) -> tuple[list[int], list[float]]:
        """openai/whisper's beam search: keep the `beam_size` most likely texts so far; each step every one is
        extended by its most likely next tokens and the best `beam_size` of all those are kept. A text that ends
        (<|endoftext|>) while ranking above the last one kept is set aside, until `beam_size` have ended. The winner has the best mean log-probability
        per token (longer texts aren't penalized for being longer)."""
        eot = self.tokens.eot
        # (sampled tokens, their log-probabilities, sum, decoder session holding the K/V of prompt + sampled)
        beams = [([], [], 0.0, self._decoder_session(encoded, kv_cache))]
        finished: list[tuple[list[int], list[float], float]] = []
        for _ in range(steps):
            candidates = []
            for sampled, lps, total, step in beams:
                logits = step(prompt + sampled)
                self._apply_rules(logits, sampled)
                finite = np.isfinite(logits)
                log_probs = np.where(finite, logits - np.logaddexp.reduce(logits[finite]), -np.inf)
                for token in np.argsort(log_probs)[-(beam_size + 1):]:
                    lp = float(log_probs[token])
                    if np.isfinite(lp):
                        candidates.append((total + lp, int(token), lp, sampled, lps, step))
            candidates.sort(key=lambda c: c[0], reverse=True)
            beams = []
            for total, token, lp, sampled, lps, step in candidates:  # best first, until the beams are full
                if token != eot:
                    beams.append((sampled + [token], lps + [lp], total, step.fork()))
                    if len(beams) == beam_size:
                        break
                elif len(finished) < beam_size:  # an ended text that ranks among the best of this step
                    finished.append((sampled + [token], lps + [lp], total))
            if len(finished) >= beam_size or not beams:
                break
        pool = finished or [(sampled, lps, total) for sampled, lps, total, _ in beams]
        best = max(pool, key=lambda b: b[2] / max(len(b[0]), 1))
        return best[0], best[1]

    def _apply_rules(self, logits: np.ndarray, sampled: list[int]) -> None:
        """Whisper's logit filters: suppress specials and enforce well-formed timestamp pairs."""
        t = self.tokens
        ts = t.timestamp_begin
        logits[t.suppress] = -np.inf
        # Special tokens between <|endoftext|> and the timestamps are never valid output.
        logits[t.eot + 1 : ts] = -np.inf

        if not sampled:
            # First token must be a timestamp near the window start.
            logits[: ts] = -np.inf
            logits[ts + int(MAX_INITIAL_TIMESTAMP / TIME_PRECISION) + 1 :] = -np.inf
            return

        last_was_ts = sampled[-1] >= ts
        penultimate_was_ts = len(sampled) < 2 or sampled[-2] >= ts
        if last_was_ts:
            if penultimate_was_ts:
                logits[ts:] = -np.inf  # a segment needs text after its start timestamp
            else:
                logits[: t.eot] = -np.inf  # after closing a segment: timestamp or end

        timestamps = [tok for tok in sampled if tok >= ts]
        if timestamps:
            # Timestamps never go backwards (a closing tag may be repeated as the next opening).
            floor = timestamps[-1] if last_was_ts and not penultimate_was_ts else timestamps[-1] + 1
            logits[ts:floor] = -np.inf

        # If any timestamp is more likely than all text tokens combined, emit a timestamp.
        log_probs = logits - np.logaddexp.reduce(logits[np.isfinite(logits)])
        ts_mass = np.logaddexp.reduce(log_probs[ts:]) if np.isfinite(log_probs[ts:]).any() else -np.inf
        if ts_mass > np.max(log_probs[:ts]):
            logits[:ts] = -np.inf

    def _split_segments(self, sampled: list[int], window_seconds: float,
                        logprobs: list[float] | None = None) -> tuple[list[Segment], float]:
        t = self.tokens
        if sampled and sampled[-1] == t.eot:
            sampled = sampled[:-1]
            finished = True
        else:
            finished = False  # hit max_new_tokens

        logprobs = logprobs or [0.0] * len(sampled)
        segments: list[Segment] = []
        start: float | None = None
        text_ids: list[int] = []
        text_lps: list[float] = []
        last_end = 0.0

        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        for tok, lp in zip(sampled, logprobs):
            if tok >= t.timestamp_begin:
                time = (tok - t.timestamp_begin) * TIME_PRECISION
                if start is None:
                    start, text_ids, text_lps = time, [], []
                else:
                    segments.append(Segment(start, time, self._decode_text(text_ids), mean(text_lps),
                                            tokens=text_ids))
                    last_end, start = time, None
            elif start is not None:
                text_ids.append(tok)
                text_lps.append(lp)

        if start is not None and text_ids:
            # Unclosed trailing segment: keep it if the model finished, otherwise re-decode from there.
            if finished or not segments:
                segments.append(Segment(start, window_seconds, self._decode_text(text_ids), mean(text_lps),
                                        tokens=text_ids))
                return segments, window_seconds
            return segments, last_end

        if finished or not segments:
            return segments, window_seconds
        return segments, last_end

    def _decode_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
