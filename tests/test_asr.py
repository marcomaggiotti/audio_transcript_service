"""Whisper decoding logic, driven by scripted stand-ins for the ONNX sessions."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.asr import DecodingOptions, ModelLoadError, OnnxWhisperTranscriber, WhisperTokens

# Tiny vocabulary: 0-9 text, then specials, then timestamps (0.02 s each).
WORDS = ["hello", "world", "onnx", "runtime", "bye", "x5", "x6", "x7", "x8", "x9"]
EOT, SOT, NO_TS = 10, 11, 15
PREV = 9  # stands in for <|startofprev|>; the scripted decoder only needs SOT to find the prompt
TS = 16
VOCAB = TS + 1501


def ts(seconds: float) -> int:
    return TS + round(seconds / 0.02)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(WORDS[i] for i in ids if i < 10)

    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=[WORDS.index(w) for w in text.split() if w in WORDS])


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input_features")]

    def run(self, outputs, feeds):
        self.calls += 1
        assert feeds["input_features"].shape == (1, 80, 3000)
        return [np.full((1, 1500, 4), self.calls, dtype=np.float32)]


class ScriptedDecoder:
    """Returns logits that prefer the next token of a per-window script."""

    def __init__(self, scripts, sot_scores=None, strength=10.0):
        self.scripts = scripts
        self.sot_scores = sot_scores or {}  # logits of the step right after <|startoftranscript|> (no-speech odds)
        self.strength = strength  # logit of the scripted token against zeros (lower = less sure)
        self.prompts: list[list[int]] = []  # every decoding attempt's prompt
        self.attempts: dict[int, int] = {}
        self.windows_seen: set[int] = set()

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="encoder_hidden_states")]

    def run(self, outputs, feeds):
        ids = feeds["input_ids"][0].tolist()
        window = int(feeds["encoder_hidden_states"][0, 0, 0]) - 1
        logits = np.zeros(VOCAB, dtype=np.float32)
        # The English-only prompt is just <|startoftranscript|>, the same tokens as the no-speech step, which is the
        # first decoder call of every window.
        if ids == [SOT] and window not in self.windows_seen:
            self.windows_seen.add(window)
            for token, score in self.sot_scores.items():
                logits[token] = score
        else:
            prompt = ids[: ids.index(SOT) + 1]  # previous-text prompt included
            step = len(ids) - len(prompt)
            if step == 0:  # first step of a decoding attempt
                self.prompts.append(prompt)
                self.attempts[window] = self.attempts.get(window, -1) + 1
            script = self.scripts[window]
            if script and isinstance(script[0], list):  # one script per fallback attempt
                script = script[min(self.attempts[window], len(script) - 1)]
            logits[script[step] if step < len(script) else EOT] = self.strength
        return [logits[None, None, :]]


def make(scripts, **kw):
    tokens = WhisperTokens(eot=EOT, sot=SOT, no_timestamps=NO_TS, timestamp_begin=TS, suppress=[], prev=PREV)
    decoder = ScriptedDecoder(scripts, **kw)
    return OnnxWhisperTranscriber(FakeEncoder(), decoder, FakeTokenizer(), tokens)


def test_segments_with_timestamps():
    script = [ts(0), 0, 1, ts(2.0), ts(2.0), 2, 3, ts(4.5), EOT]
    result = make([script]).transcribe(np.zeros(16000 * 5, dtype=np.float32))
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 2.0, "hello world"), (2.0, 4.5, "onnx runtime"),
    ]
    assert result.text == "hello world onnx runtime"


def test_long_audio_is_split_into_windows_with_offsets():
    first = [ts(0), 0, ts(29.0), EOT]  # single closing timestamp: whole window consumed
    second = [ts(1.0), 4, ts(3.0), EOT]
    result = make([first, second]).transcribe(np.zeros(16000 * 40, dtype=np.float32))
    assert [(s.start, s.end, s.text) for s in result.segments] == [(0.0, 29.0, "hello"), (31.0, 33.0, "bye")]
    # The trace: one closing timestamp consumes the whole window, so the second starts at 30 s with the last 10 s.
    assert [(w.start, w.length, w.consumed, w.segments, w.skipped) for w in result.windows] == [
        (0.0, 30.0, 30.0, 1, False), (30.0, 10.0, 10.0, 1, False)]
    assert [a["temperature"] for a in result.windows[0].attempts] == [0.0]


def test_end_times_are_clamped_to_audio_duration():
    result = make([[ts(0), 0, ts(10.0), EOT]]).transcribe(np.zeros(16000 * 3, dtype=np.float32))
    assert result.segments[0].end == 3.0


def test_timestamp_rules_block_bad_output():
    # The script asks for text first and a timestamp going backwards; the rules must override both.
    script = [0, ts(0.5), 1, ts(0.2)]
    result = make([script]).transcribe(np.zeros(16000 * 2, dtype=np.float32))
    seg = result.segments[0]
    assert seg.start == 0.0  # first token forced to a (near-zero) timestamp
    assert seg.end >= seg.start


def test_prompt_is_start_of_transcript():
    t = make([[ts(0), 0, ts(1.0), EOT]])
    assert t._prompt() == [SOT]
    assert t._prompt([0, 1]) == [PREV, 0, 1, SOT]  # after the previous text
    result = t.transcribe(np.zeros(16000, dtype=np.float32))
    assert result.segments[0].text == "hello"


def test_missing_model_dir(tmp_path):
    with pytest.raises(ModelLoadError, match="export_whisper_onnx"):
        OnnxWhisperTranscriber.from_dir(str(tmp_path))


def test_confidence_and_no_speech_are_recorded():
    NOSPEECH = 9  # any id works for the scripted decoder
    OTHER = 12  # some other token, more likely than no-speech
    t = make([[ts(0), 0, 1, ts(2.0), EOT]], sot_scores={OTHER: 3.0, NOSPEECH: 2.0})
    t.tokens.no_speech = NOSPEECH
    [seg] = t.transcribe(np.zeros(16000 * 3, dtype=np.float32)).segments
    sot = np.zeros(VOCAB)
    sot[OTHER], sot[NOSPEECH] = 3.0, 2.0
    assert seg.no_speech_prob == pytest.approx(np.exp(2.0) / np.exp(sot).sum(), abs=1e-4)
    # Each text token had logit 10 against ~1500 zeros (fewer after the timestamp rules).
    assert -0.2 < seg.avg_logprob < 0


def test_progress_after_each_window():
    first = [ts(0), 0, ts(29.0), EOT]  # a single closing timestamp: the whole 30 s window is consumed
    second = [ts(1.0), 4, ts(3.0), EOT]
    seen = []
    make([first, second]).transcribe(np.zeros(16000 * 40, dtype=np.float32), on_progress=seen.append)
    assert seen[0] == pytest.approx(30 / 40) and seen[-1] == 1.0


def test_greedy_by_default_decodes_each_window_once():
    t = make([[ts(0), 0, ts(1.0), EOT]])
    t.transcribe(np.zeros(16000 * 2, dtype=np.float32))
    assert t.decoder.attempts == {0: 0}


def test_temperature_fallback_redecodes_a_repetition_loop():
    loop = [ts(0)] + [0] * 60 + [ts(5.0), EOT]  # "hello hello hello ...": compresses far too well
    good = [ts(0), 2, 3, ts(5.0), EOT]
    audio = np.zeros(16000 * 6, dtype=np.float32)

    greedy = make([[loop, good]]).transcribe(audio)
    assert greedy.segments[0].text.startswith("hello hello")

    t = make([[loop, good]])
    fixed = t.transcribe(audio, options=DecodingOptions(temperature_fallback=True))
    assert [s.text for s in fixed.segments] == ["onnx runtime"]
    assert t.decoder.attempts == {0: 1}  # second attempt (temperature 0.2) was good enough
    [window] = fixed.windows
    assert [a["temperature"] for a in window.attempts] == [0.0, 0.2] and window.attempts[0]["compression_ratio"] > 2.4


def test_fallback_skips_silent_unsure_windows():
    NOSPEECH = 8
    audio = np.zeros(16000 * 3, dtype=np.float32)
    options = DecodingOptions(temperature_fallback=True)
    # Weak logits make every token unsure (avg log-prob well below -1); no-speech dominates the first step.
    t = make([[ts(0), 0, ts(1.0), EOT]], sot_scores={NOSPEECH: 10.0}, strength=4.0)
    t.tokens.no_speech = NOSPEECH
    silent = t.transcribe(audio, options=options)
    assert silent.segments == []
    assert t.decoder.attempts == {0: 0}  # silence: no retries
    assert silent.windows[0].skipped and silent.windows[0].no_speech_prob > 0.6 and silent.windows[0].segments == 0
    t = make([[ts(0), 0, ts(1.0), EOT]], sot_scores={NOSPEECH: 10.0}, strength=4.0)
    t.tokens.no_speech = NOSPEECH
    assert [s.text for s in t.transcribe(audio).segments] == ["hello"]  # greedy keeps it


def test_previous_text_is_prompted_on_the_next_window():
    first = [ts(0), 0, 1, ts(29.0), EOT]
    second = [ts(0), 2, ts(3.0), EOT]
    t = make([first, second])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32),
                 options=DecodingOptions(condition_on_previous_text=True))
    assert t.decoder.prompts == [[SOT], [PREV, 0, 1, SOT]]

    t = make([first, second])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32))
    assert t.decoder.prompts == [[SOT]] * 2


def test_initial_prompt_on_every_window():
    t = make([[ts(0), 2, ts(29.0), EOT], [ts(0), 4, ts(2.0), EOT]])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32),
                 options=DecodingOptions(initial_prompt="onnx runtime", condition_on_previous_text=True))
    assert t.decoder.prompts == [[PREV, 2, 3, SOT], [PREV, 2, 3, 2, SOT]]


def test_onnx_precision_reads_the_graph_ops(tmp_path):
    from app.asr import onnx_precision
    (tmp_path / "q.onnx").write_bytes(b"\x08\x07...DynamicQuantizeLinear...MatMulInteger...")
    (tmp_path / "f.onnx").write_bytes(b"\x08\x07...MatMul...Softmax...")
    assert onnx_precision(tmp_path / "q.onnx") == "int8" and onnx_precision(tmp_path / "f.onnx") == "fp32"


def test_precision_swaps_parts_for_one_job_and_back(tmp_path):
    t = make([[ts(0), 0, 1, ts(2.0), EOT]] * 5)  # FakeEncoder numbers windows across calls
    default_encoder = t.encoder
    t.precision = {"encoder": "fp32", "decoder": "int8"}
    t.variants = {"encoder": {"fp32": tmp_path / "e.onnx", "int8": tmp_path / "e8.onnx"},
                  "decoder": {"int8": tmp_path / "d.onnx", "fp32": tmp_path / "d32.onnx"}}
    loaded = []

    def load(path):
        loaded.append(path.name)
        if path.name.startswith("e"):
            return FakeEncoder()
        words = (2, 3) if path.name == "d32.onnx" else (0, 1)  # the fp32 "model" writes different words
        return ScriptedDecoder([[ts(0), *words, ts(2.0), EOT]] * 5)

    t._load_session = load
    assert t.available_precisions == {"encoder": ["fp32", "int8"], "decoder": ["fp32", "int8"]}
    with t.precision_for("fp32", None) as used:  # the default: nothing loaded
        assert used == {"encoder": "fp32", "decoder": "int8"} and loaded == []
    with t.precision_for(None, "fp32") as used:
        assert used == {"encoder": "fp32", "decoder": "fp32"} and t.encoder is default_encoder
        assert t.transcribe(np.zeros(16000 * 3, dtype=np.float32)).text == "onnx runtime"
    assert loaded == ["d32.onnx", "d.onnx"]  # swapped in, then the default loaded back
    assert t.transcribe(np.zeros(16000 * 3, dtype=np.float32)).text == "hello world"
    with pytest.raises(ValueError, match="encoder precision fp16 isn't available"):
        with t.precision_for("fp16", None):
            pass


def test_only_english_only_models_load():
    class Tokenizer:  # whisper-tiny.en's special tokens (an .en tokenizer still has language tokens)
        ids = {"<|endoftext|>": 50256, "<|startoftranscript|>": 50257, "<|en|>": 50258, "<|it|>": 50274,
               "<|transcribe|>": 50358, "<|startofprev|>": 50360, "<|nocaptions|>": 50361}

        def token_to_id(self, token):
            return self.ids.get(token)

    from app.asr import _resolve_tokens
    english = _resolve_tokens(Tokenizer(), {"is_multilingual": False, "no_timestamps_token_id": 50362})
    assert (english.sot, english.timestamp_begin, english.no_speech, english.prev) == (50257, 50363, 50361, 50360)
    # The English-only flag wins even if the config also lists language tokens.
    assert _resolve_tokens(Tokenizer(), {"is_multilingual": False, "no_timestamps_token_id": 50362,
                                         "lang_to_id": {"<|en|>": 50258}}).sot == 50257
    for multilingual in ({"is_multilingual": True}, {"lang_to_id": {"<|en|>": 50259, "<|it|>": 50274}}):
        with pytest.raises(ModelLoadError, match="multilingual"):
            _resolve_tokens(Tokenizer(), {"no_timestamps_token_id": 50363, **multilingual})


class MergedDecoder(ScriptedDecoder):
    """Stand-in for optimum's merged decoder: use_cache_branch picks the cached branch, and the "keys" it caches
    are the token ids themselves, so a step sees past + new tokens exactly when the cache is carried right."""

    def __init__(self, scripts, **kw):
        super().__init__(scripts, **kw)
        self.calls: list[tuple[bool, int]] = []  # (cached branch, tokens fed) of every call

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input_ids", "encoder_hidden_states", "use_cache_branch")] + [
            SimpleNamespace(name=f"past_key_values.0.{part}.key", shape=["batch", 1, "seq", 1])
            for part in ("decoder", "encoder")]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ("logits", "present.0.decoder.key", "present.0.encoder.key")]

    def run(self, outputs, feeds):
        cached = bool(feeds["use_cache_branch"][0])
        new = feeds["input_ids"][0].tolist()
        past = feeds["past_key_values.0.decoder.key"].reshape(-1).astype(int).tolist() if cached else []
        self.calls.append((cached, len(new)))
        ids = past + new
        [logits] = super().run(None, {"input_ids": np.array([ids]), "encoder_hidden_states": feeds["encoder_hidden_states"]})
        produced = {"logits": logits, "present.0.decoder.key": np.array(ids, np.float32).reshape(1, 1, -1, 1),
                    "present.0.encoder.key": np.zeros((1, 1, 1500, 1), np.float32)}
        return [produced[name] for name in (outputs or [o.name for o in self.get_outputs()])]


@pytest.mark.parametrize("beam_size", [1, 3])
def test_kv_cache_changes_the_work_not_the_transcript(beam_size):
    scripts = [[ts(0), 0, 1, ts(2.0), ts(2.0), 2, 3, ts(29.0), EOT], [ts(1.0), 4, ts(3.0), EOT]]
    audio = np.zeros(16000 * 40, dtype=np.float32)
    results, calls = {}, {}
    for kv_cache in (True, False):
        decoder = MergedDecoder(scripts)
        t = OnnxWhisperTranscriber(FakeEncoder(), decoder, FakeTokenizer(), make([]).tokens)
        result = t.transcribe(audio, options=DecodingOptions(kv_cache=kv_cache, beam_size=beam_size,
                                                                   condition_on_previous_text=True))
        results[kv_cache] = [(s.start, s.end, s.text, s.avg_logprob) for s in result.segments]
        calls[kv_cache] = [c for c in decoder.calls if c != (False, 1)]  # minus the one-token no-speech steps
    assert results[True] == results[False]
    assert results[True][0][2] == "hello world" and results[True][-1][2] == "bye"
    # Cached: after the prompt, one new token per call. Off: always the first-call branch, with the whole text.
    assert all(n == 1 for cached, n in calls[True] if cached) and any(cached for cached, _ in calls[True])
    assert not any(cached for cached, _ in calls[False]) and max(n for _, n in calls[False]) > 5


def test_cancel_check_stops_decoding_between_steps():
    from app.asr import Cancelled, cancel_check

    t = make([[ts(0)] + [0] * 100 + [ts(5.0), EOT]])
    steps = []

    def check():
        steps.append(1)
        if len(steps) > 5:
            raise Cancelled("stop")

    token = cancel_check.set(check)
    try:
        with pytest.raises(Cancelled):
            t.transcribe(np.zeros(16000 * 6, dtype=np.float32))
    finally:
        cancel_check.reset(token)
    assert len(steps) == 6  # checked before each decoder step, stopped at the first one after cancelling
    fresh = make([[ts(0)] + [0] * 100 + [ts(5.0), EOT]])
    assert fresh.transcribe(np.zeros(16000 * 6, dtype=np.float32)).segments  # no check set: runs to the end
