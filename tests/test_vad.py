"""Voice activity detection before Whisper (app/vad.py), with a scripted VAD instead of the Silero model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr import DecodingOptions, Segment, TranscriptionResult, WindowTrace
from app.main import Components, create_app
from app.vad import FRAME, pieces, speech_spans, transcribe_speech, transcribe_with
from tests.conftest import FakeTranscriber, make_wav, wait_for

SR = 16000


def probs_for(regions, seconds):
    """Frame probabilities: 0.9 inside the (start, end) regions in seconds, 0.05 elsewhere."""
    n = -(-int(seconds * SR) // FRAME)
    t = np.arange(n) * FRAME / SR
    p = np.full(n, 0.05, dtype=np.float32)
    for a, b in regions:
        p[(t >= a) & (t < b)] = 0.9
    return p


def test_speech_spans_follow_the_speech_with_padding():
    spans = speech_spans(probs_for([(1.0, 3.0), (6.0, 8.0)], 10), 10 * SR, min_silence_ms=500, speech_pad_ms=200)
    assert [(round(a, 1), round(b, 1)) for a, b in spans] == [(0.8, 3.2), (5.8, 8.2)]


def test_short_pauses_join_and_blips_are_dropped():
    spans = speech_spans(probs_for([(1.0, 3.0), (3.2, 5.0), (7.0, 7.1)], 10), 10 * SR, min_silence_ms=500)
    assert len(spans) == 1 and round(spans[0][1], 1) == 5.2  # 0.2 s pause kept inside, 0.1 s blip gone


def test_long_speech_is_cut_at_a_pause_before_30_seconds():
    regions = [(k * 4.0, k * 4.0 + 3.8) for k in range(15)]  # 60 s of speech with 0.2 s pauses
    spans = speech_spans(probs_for(regions, 60), 60 * SR, min_silence_ms=500)
    assert len(spans) >= 3 and all(b - a <= 29.0 for a, b in spans)


def test_pieces_hold_at_most_29_seconds_of_speech():
    spans = [(0, 10), (12, 22), (25, 35), (40, 45)]
    assert pieces(spans) == [[(0, 10), (12, 22)], [(25, 35), (40, 45)]]


class ScriptedVad:
    def __init__(self, spans):
        self.spans, self.calls = spans, []

    def speech_spans(self, audio, threshold=0.5, min_silence_ms=500):
        self.calls.append((threshold, min_silence_ms))
        return self.spans


class PieceTranscriber:
    """Says how long each piece it hears is, in one segment over the whole piece."""

    name = "fake"

    def __init__(self):
        self.lengths, self.prompts = [], []

    def transcribe(self, audio, language=None, on_progress=None, options=None):
        seconds = round(audio.shape[0] / SR, 2)
        self.lengths.append(seconds)
        self.prompts.append(options.initial_prompt)
        text = "hello " * 40 if seconds == 4.0 else f"piece of {seconds}"
        return TranscriptionResult(language=language or "it", segments=[Segment(0.0, seconds, text.strip(), -0.2)],
                                   windows=[WindowTrace(0.0, seconds, seconds, 0.01, False, 1)])


def test_only_the_speech_is_transcribed_and_times_map_back():
    audio = np.zeros(60 * SR, dtype=np.int16)
    vad = ScriptedVad([(2.0, 12.0), (20.0, 30.0), (31.0, 41.0), (50.0, 54.0)])
    t = PieceTranscriber()
    shares = []
    result = transcribe_speech(t, vad, audio, None, shares.append,
                               DecodingOptions(vad=True, vad_threshold=0.6, vad_min_silence_ms=800))
    assert vad.calls == [(0.6, 800)]
    assert t.lengths == [20.0, 14.0]  # silence left out: 10 s + 10 s, then 10 s + 4 s
    # A piece's segment covers its speech: from its first stretch's start to its last one's end.
    assert [(s.start, s.end) for s in result.segments] == [(2.0, 30.0), (31.0, 54.0)]
    assert [w.start for w in result.windows] == [2.0, 31.0]
    assert result.speech == vad.spans and result.language == "it" and shares[-1] == 1.0


def test_no_speech_no_transcription():
    t = PieceTranscriber()
    result = transcribe_speech(t, ScriptedVad([]), np.zeros(10 * SR, dtype=np.int16), "en", None,
                               DecodingOptions(vad=True))
    assert t.lengths == [] and result.segments == [] and result.speech == []


def test_previous_text_carries_between_pieces():
    t = PieceTranscriber()
    vad = ScriptedVad([(0.0, 20.0), (25.0, 30.0), (35.0, 45.0)])  # 20 + 5 s, then 10 s
    transcribe_speech(t, vad, np.zeros(60 * SR, dtype=np.int16), "en", None,
                      DecodingOptions(vad=True, condition_on_previous_text=True, initial_prompt="Aipollo"))
    assert t.lengths == [25.0, 10.0]
    assert t.prompts == ["Aipollo", "Aipollo piece of 25.0"]


def test_a_looping_piece_is_not_prompted_to_the_next():
    audio = np.zeros(80 * SR, dtype=np.int16)
    t = PieceTranscriber()
    vad = ScriptedVad([(0.0, 4.0), (30.0, 58.0), (60.0, 70.0)])  # 4 s (loops) + nothing fits after: 28 s piece
    result = transcribe_speech(t, vad, audio, "en", None, DecodingOptions(vad=True, condition_on_previous_text=True))
    assert t.lengths == [4.0, 28.0, 10.0]
    assert t.prompts == [None, None, "piece of 28.0"]
    assert result.windows[0].prompt_reset


def test_without_vad_the_transcriber_hears_everything():
    t = PieceTranscriber()
    transcribe_with(t, np.zeros(10 * SR, dtype=np.int16), options=DecodingOptions())
    assert t.lengths == [10.0]
    with pytest.raises(RuntimeError, match="VAD model"):
        transcribe_with(t, np.zeros(10 * SR, dtype=np.int16), options=DecodingOptions(vad=True))


def test_jobs_take_vad_and_beam_options(settings):
    transcriber = FakeTranscriber()
    vad = ScriptedVad([(0.2, 0.8)])
    with TestClient(create_app(settings, Components(transcriber=transcriber, vad=vad))) as client:
        assert client.get("/config").json()["decoding"]["vad"]["available"] is True
        files = {"file": ("a.wav", make_wav(1.0), "audio/wav")}
        job = client.post("/jobs", files=files, data={"vad": "true", "vad_threshold": "0.6", "beam_size": "5",
                                                        "reset_on_loop": "false"}).json()
        body = wait_for(client, job["id"])
        assert body["status"] == "completed" and body["vad"] and body["beam_size"] == 5
        options = transcriber.options[-1]
        assert (options.beam_size, options.reset_on_loop, options.vad, options.vad_threshold) == (5, False, True, 0.6)
        assert vad.calls == [(0.6, 500)]
        transcript = client.get(f"/jobs/{job['id']}/transcript").json()
        assert transcript["speech"] == [[0.2, 0.8]]
        assert client.post("/jobs", files=files, data={"beam_size": "9"}).status_code == 422


def test_vad_asked_for_without_the_model(client, transcriber):
    response = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")}, data={"vad": "true"})
    assert response.status_code == 503 and "vad=false" in response.json()["detail"]
    assert client.get("/config").json()["decoding"]["vad"]["available"] is False
