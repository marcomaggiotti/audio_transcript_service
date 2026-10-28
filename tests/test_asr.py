"""Whisper decoding logic, driven by scripted stand-ins for the ONNX sessions."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.asr import DecodingOptions, ModelLoadError, OnnxWhisperTranscriber, WhisperTokens

# Tiny vocabulary: 0-9 text, then specials, then timestamps (0.02 s each).
WORDS = ["hello", "world", "onnx", "runtime", "bye", "x5", "x6", "x7", "x8", "x9"]
EOT, SOT, EN, DE, TRANSCRIBE, NO_TS = 10, 11, 12, 13, 14, 15
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

    def __init__(self, scripts, prompt_len=3, language_scores=None, strength=10.0):
        self.scripts = scripts
        self.prompt_len = prompt_len
        self.language_scores = language_scores or {}
        self.strength = strength  # logit of the scripted token against zeros (lower = less sure)
        self.prompts: list[list[int]] = []  # every decoding attempt's prompt
        self.attempts: dict[int, int] = {}

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="encoder_hidden_states")]

    def run(self, outputs, feeds):
        ids = feeds["input_ids"][0].tolist()
        window = int(feeds["encoder_hidden_states"][0, 0, 0]) - 1
        logits = np.zeros(VOCAB, dtype=np.float32)
        if ids == [SOT]:  # language detection
            for token, score in self.language_scores.items():
                logits[token] = score
        else:
            prompt = ids[: ids.index(SOT) + self.prompt_len]  # previous-text prompt included
            step = len(ids) - len(prompt)
            if step == 0:  # first step of a decoding attempt
                self.prompts.append(prompt)
                self.attempts[window] = self.attempts.get(window, -1) + 1
            script = self.scripts[window]
            if script and isinstance(script[0], list):  # one script per fallback attempt
                script = script[min(self.attempts[window], len(script) - 1)]
            logits[script[step] if step < len(script) else EOT] = self.strength
        return [logits[None, None, :]]


def make(scripts, multilingual=True, **kw):
    tokens = WhisperTokens(
        eot=EOT, sot=SOT, transcribe=TRANSCRIBE, no_timestamps=NO_TS, timestamp_begin=TS,
        lang_to_id={"en": EN, "de": DE} if multilingual else {}, suppress=[], prev=PREV,
    )
    decoder = ScriptedDecoder(scripts, prompt_len=3 if multilingual else 1, **kw)
    return OnnxWhisperTranscriber(FakeEncoder(), decoder, FakeTokenizer(), tokens)


def test_segments_with_timestamps():
    script = [ts(0), 0, 1, ts(2.0), ts(2.0), 2, 3, ts(4.5), EOT]
    result = make([script]).transcribe(np.zeros(16000 * 5, dtype=np.float32), language="en")
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 2.0, "hello world"), (2.0, 4.5, "onnx runtime"),
    ]
    assert result.text == "hello world onnx runtime"
    assert result.language == "en"


def test_long_audio_is_split_into_windows_with_offsets():
    first = [ts(0), 0, ts(29.0), EOT]  # single closing timestamp: whole window consumed
    second = [ts(1.0), 4, ts(3.0), EOT]
    result = make([first, second]).transcribe(np.zeros(16000 * 40, dtype=np.float32), language="en")
    assert [(s.start, s.end, s.text) for s in result.segments] == [(0.0, 29.0, "hello"), (31.0, 33.0, "bye")]
    # The trace: one closing timestamp consumes the whole window, so the second starts at 30 s with the last 10 s.
    assert [(w.start, w.length, w.consumed, w.segments, w.skipped) for w in result.windows] == [
        (0.0, 30.0, 30.0, 1, False), (30.0, 10.0, 10.0, 1, False)]
    assert [a["temperature"] for a in result.windows[0].attempts] == [0.0]


def test_end_times_are_clamped_to_audio_duration():
    result = make([[ts(0), 0, ts(10.0), EOT]]).transcribe(np.zeros(16000 * 3, dtype=np.float32), "en")
    assert result.segments[0].end == 3.0


def test_timestamp_rules_block_bad_output():
    # The script asks for text first and a timestamp going backwards; the rules must override both.
    script = [0, ts(0.5), 1, ts(0.2)]
    result = make([script]).transcribe(np.zeros(16000 * 2, dtype=np.float32), "en")
    seg = result.segments[0]
    assert seg.start == 0.0  # first token forced to a (near-zero) timestamp
    assert seg.end >= seg.start


def test_language_detection():
    t = make([[ts(0), 0, ts(1.0), EOT]], language_scores={EN: 1.0, DE: 5.0})
    result = t.transcribe(np.zeros(16000 * 2, dtype=np.float32), language="auto")
    assert result.language == "de"


def test_english_only_model_uses_short_prompt():
    t = make([[ts(0), 0, ts(1.0), EOT]], multilingual=False)
    assert t._prompt("de") == [SOT]
    assert t.transcribe(np.zeros(16000, dtype=np.float32)).segments[0].text == "hello"


def test_unknown_language_rejected():
    with pytest.raises(ValueError, match="Unsupported language"):
        make([[EOT]])._prompt("xx")


def test_missing_model_dir(tmp_path):
    with pytest.raises(ModelLoadError, match="export_whisper_onnx"):
        OnnxWhisperTranscriber.from_dir(str(tmp_path))


def test_confidence_and_no_speech_are_recorded():
    NOSPEECH = 9  # any id works for the scripted decoder
    t = make([[ts(0), 0, 1, ts(2.0), EOT]], language_scores={EN: 3.0, NOSPEECH: 2.0})
    t.tokens.no_speech = NOSPEECH
    [seg] = t.transcribe(np.zeros(16000 * 3, dtype=np.float32), language="auto").segments
    sot = np.zeros(VOCAB)
    sot[EN], sot[NOSPEECH] = 3.0, 2.0
    assert seg.no_speech_prob == pytest.approx(np.exp(2.0) / np.exp(sot).sum(), abs=1e-4)
    # Each text token had logit 10 against ~1500 zeros (fewer after the timestamp rules).
    assert -0.2 < seg.avg_logprob < 0


def test_progress_after_each_window():
    first = [ts(0), 0, ts(29.0), EOT]  # a single closing timestamp: the whole 30 s window is consumed
    second = [ts(1.0), 4, ts(3.0), EOT]
    seen = []
    make([first, second]).transcribe(np.zeros(16000 * 40, dtype=np.float32), "en", on_progress=seen.append)
    assert seen[0] == pytest.approx(30 / 40) and seen[-1] == 1.0


def test_greedy_by_default_decodes_each_window_once():
    t = make([[ts(0), 0, ts(1.0), EOT]])
    t.transcribe(np.zeros(16000 * 2, dtype=np.float32), "en")
    assert t.decoder.attempts == {0: 0}


def test_temperature_fallback_redecodes_a_repetition_loop():
    loop = [ts(0)] + [0] * 60 + [ts(5.0), EOT]  # "hello hello hello ...": compresses far too well
    good = [ts(0), 2, 3, ts(5.0), EOT]
    audio = np.zeros(16000 * 6, dtype=np.float32)

    greedy = make([[loop, good]]).transcribe(audio, "en")
    assert greedy.segments[0].text.startswith("hello hello")

    t = make([[loop, good]])
    fixed = t.transcribe(audio, "en", options=DecodingOptions(temperature_fallback=True))
    assert [s.text for s in fixed.segments] == ["onnx runtime"]
    assert t.decoder.attempts == {0: 1}  # second attempt (temperature 0.2) was good enough
    [window] = fixed.windows
    assert [a["temperature"] for a in window.attempts] == [0.0, 0.2] and window.attempts[0]["compression_ratio"] > 2.4


def test_fallback_skips_silent_unsure_windows():
    NOSPEECH = 8
    audio = np.zeros(16000 * 3, dtype=np.float32)
    options = DecodingOptions(temperature_fallback=True)
    # Weak logits make every token unsure (avg log-prob well below -1); no-speech dominates the first step.
    t = make([[ts(0), 0, ts(1.0), EOT]], language_scores={NOSPEECH: 10.0}, strength=4.0)
    t.tokens.no_speech = NOSPEECH
    silent = t.transcribe(audio, "en", options=options)
    assert silent.segments == []
    assert t.decoder.attempts == {0: 0}  # silence: no retries
    assert silent.windows[0].skipped and silent.windows[0].no_speech_prob > 0.6 and silent.windows[0].segments == 0
    t = make([[ts(0), 0, ts(1.0), EOT]], language_scores={NOSPEECH: 10.0}, strength=4.0)
    t.tokens.no_speech = NOSPEECH
    assert [s.text for s in t.transcribe(audio, "en").segments] == ["hello"]  # greedy keeps it


def test_previous_text_is_prompted_on_the_next_window():
    first = [ts(0), 0, 1, ts(29.0), EOT]
    second = [ts(0), 2, ts(3.0), EOT]
    t = make([first, second])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32), "en",
                 options=DecodingOptions(condition_on_previous_text=True))
    assert t.decoder.prompts == [[SOT, EN, TRANSCRIBE], [PREV, 0, 1, SOT, EN, TRANSCRIBE]]

    t = make([first, second])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32), "en")
    assert t.decoder.prompts == [[SOT, EN, TRANSCRIBE]] * 2


def test_initial_prompt_on_every_window():
    t = make([[ts(0), 2, ts(29.0), EOT], [ts(0), 4, ts(2.0), EOT]])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32), "en",
                 options=DecodingOptions(initial_prompt="onnx runtime", condition_on_previous_text=True))
    assert t.decoder.prompts == [[PREV, 2, 3, SOT, EN, TRANSCRIBE], [PREV, 2, 3, 2, SOT, EN, TRANSCRIBE]]


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
        assert t.transcribe(np.zeros(16000 * 3, dtype=np.float32), "en").text == "onnx runtime"
    assert loaded == ["d32.onnx", "d.onnx"]  # swapped in, then the default loaded back
    assert t.transcribe(np.zeros(16000 * 3, dtype=np.float32), "en").text == "hello world"
    with pytest.raises(ValueError, match="encoder precision fp16 isn't available"):
        with t.precision_for("fp16", None):
            pass


def test_transcribe_intervals_one_segment_per_span():
    from app.asr import DecodingOptions, Segment, TranscriptionResult, WindowTrace, transcribe_intervals

    class Recorder:
        def __init__(self):
            self.lengths, self.languages = [], []

        def transcribe(self, audio, language=None, on_progress=None, options=None):
            assert options.condition_on_previous_text is False
            self.lengths.append(round(audio.shape[0] / 16000, 2))
            self.languages.append(language)
            return TranscriptionResult(language=language or "it", segments=[
                Segment(0.0, 1.0, " uno", -0.2, 0.1), Segment(1.0, 2.0, "due ", -0.4, 0.3)],
                windows=[WindowTrace(0.0, 2.0, 2.0, 0.1, False, 2)])

    rec = Recorder()
    audio = np.zeros(16000 * 10, dtype=np.int16)
    shares = []
    result = transcribe_intervals(rec, audio, [(0.0, 4.0), (4.0, 9.5), (9.5, 12.0)], None, shares.append,
                                  DecodingOptions(condition_on_previous_text=True))
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 4.0, "uno due"), (4.0, 9.5, "uno due"), (9.5, 10.0, "uno due")]  # clamped to the audio's end
    assert result.segments[0].avg_logprob == -0.3 and result.segments[0].no_speech_prob == 0.3
    assert rec.lengths == [4.0, 5.5, 0.5]
    assert rec.languages == [None, "it", "it"]  # detected once, then kept
    assert [w.start for w in result.windows] == [0.0, 4.0, 9.5]
    assert shares[-1] == 1.0
    assert result.text == "uno due uno due uno due"


# ----------------------------------------------------------------- beam search

# A first word that looks best on its own ("hello", barely ahead of "onnx") but leads to unsure words, against one
# that leads to a sure text: greedy takes the first, beam search the second.
BEAM_TABLE = {
    (): {ts(0): 10},
    (ts(0),): {0: 2.0, 2: 1.9},
    (ts(0), 0): {1: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    (ts(0), 0, 1): {ts(2.0): 10}, (ts(0), 0, 4): {ts(2.0): 10}, (ts(0), 0, 5): {ts(2.0): 10},
    (ts(0), 2): {3: 10},
    (ts(0), 2, 3): {ts(2.0): 10},
}


def table_logits(ids: list[int]) -> np.ndarray:
    logits = np.full(VOCAB, -30.0, dtype=np.float32)
    if ids == [SOT]:
        return logits
    sampled = tuple(ids[ids.index(SOT) + 3:])
    for token, value in BEAM_TABLE.get(sampled, {EOT: 10}).items():
        logits[token] = value
    return logits


class TableDecoder:
    """No KV cache: every call gets the whole prefix."""

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="encoder_hidden_states")]

    def run(self, outputs, feeds):
        return [table_logits(feeds["input_ids"][0].tolist())[None, None, :]]


class CachedTableDecoder:
    """A merged decoder whose self-attention "K/V" is the token history itself: a beam that forked wrongly (shared
    or lost cache) would see the wrong history and get the wrong logits."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="encoder_hidden_states"),
                SimpleNamespace(name="use_cache_branch"),
                SimpleNamespace(name="past_key_values.0.decoder.key", shape=[None, 1, None, 1]),
                SimpleNamespace(name="past_key_values.0.encoder.key", shape=[None, 1, None, 1])]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ("logits", "present.0.decoder.key", "present.0.encoder.key")]

    def run(self, outputs, feeds):
        self.calls += 1
        history = feeds["past_key_values.0.decoder.key"][0, 0, :, 0].astype(int).tolist()
        assert bool(feeds["use_cache_branch"][0]) == bool(history)
        ids = history + feeds["input_ids"][0].tolist()
        present = np.array(ids, dtype=np.float32)[None, None, :, None]
        return [table_logits(ids)[None, None, :], present, np.zeros((1, 1, 1, 1), dtype=np.float32)]


def make_table(decoder):
    tokens = WhisperTokens(eot=EOT, sot=SOT, transcribe=TRANSCRIBE, no_timestamps=NO_TS, timestamp_begin=TS,
                           lang_to_id={"en": EN, "de": DE}, suppress=[], prev=PREV)
    return OnnxWhisperTranscriber(FakeEncoder(), decoder, FakeTokenizer(), tokens)


@pytest.mark.parametrize("decoder", [TableDecoder, CachedTableDecoder])
def test_beam_search_finds_the_more_likely_text(decoder):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    greedy = make_table(decoder()).transcribe(audio, "en")
    assert greedy.text == "hello world"
    beam = make_table(decoder()).transcribe(audio, "en", options=DecodingOptions(beam_size=3))
    assert beam.text == "onnx runtime"
    assert [(s.start, s.end) for s in beam.segments] == [(0.0, 2.0)]
    assert beam.segments[0].avg_logprob > greedy.segments[0].avg_logprob
    assert beam.windows[0].attempts[0]["beam_size"] == 3


def test_beam_search_only_at_temperature_zero():
    loop = [ts(0)] + [0] * 60 + [ts(5.0), EOT]
    good = [ts(0), 2, 3, ts(5.0), EOT]
    t = make([[loop, good]])
    result = t.transcribe(np.zeros(16000 * 6, dtype=np.float32), "en",
                          options=DecodingOptions(temperature_fallback=True, beam_size=2))
    assert [a["beam_size"] for a in result.windows[0].attempts] == [2, 1]  # the fallback samples, as openai/whisper


# ----------------------------------------------------------------- carried text

def test_a_window_that_loops_is_not_carried_into_the_next():
    loop = [ts(0)] + [0] * 60 + [ts(29.0), EOT]
    second = [ts(0), 2, ts(3.0), EOT]
    options = DecodingOptions(condition_on_previous_text=True)
    t = make([loop, second])
    result = t.transcribe(np.zeros(16000 * 40, dtype=np.float32), "en", options=options)
    assert t.decoder.prompts[1] == [SOT, EN, TRANSCRIBE]  # started afresh
    assert result.windows[0].prompt_reset and not result.windows[1].prompt_reset

    t = make([loop, second])
    t.transcribe(np.zeros(16000 * 40, dtype=np.float32), "en",
                 options=DecodingOptions(condition_on_previous_text=True, reset_on_loop=False))
    assert t.decoder.prompts[1][:3] == [PREV, 0, 0]  # the loop is carried over
