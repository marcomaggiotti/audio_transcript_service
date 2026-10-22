"""Whisper decoding logic, driven by scripted stand-ins for the ONNX sessions."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.asr import ModelLoadError, OnnxWhisperTranscriber, WhisperTokens

# Tiny vocabulary: 0-9 text, then specials, then timestamps (0.02 s each).
WORDS = ["hello", "world", "onnx", "runtime", "bye", "x5", "x6", "x7", "x8", "x9"]
EOT, SOT, EN, DE, TRANSCRIBE, NO_TS = 10, 11, 12, 13, 14, 15
TS = 16
VOCAB = TS + 1501


def ts(seconds: float) -> int:
    return TS + round(seconds / 0.02)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(WORDS[i] for i in ids if i < 10)


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

    def __init__(self, scripts, prompt_len=3, language_scores=None):
        self.scripts = scripts
        self.prompt_len = prompt_len
        self.language_scores = language_scores or {}

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
            script = self.scripts[window]
            step = len(ids) - self.prompt_len
            logits[script[step] if step < len(script) else EOT] = 10.0
        return [logits[None, None, :]]


def make(scripts, multilingual=True, **kw):
    tokens = WhisperTokens(
        eot=EOT, sot=SOT, transcribe=TRANSCRIBE, no_timestamps=NO_TS, timestamp_begin=TS,
        lang_to_id={"en": EN, "de": DE} if multilingual else {}, suppress=[],
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
