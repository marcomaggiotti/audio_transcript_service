import io
import math
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr import Segment, TranscriptionResult
from app.config import Settings
from app.main import Components, create_app


class FakeTranscriber:
    name = "fake-whisper"
    supported_languages = ["en", "de", "it"]

    def __init__(self, segments=None, error=None):
        self.segments = segments or [
            Segment(0.0, 4.2, "Welcome to the show about machine learning.", -0.2, 0.01),
            Segment(4.2, 9.8, "Today we talk about ONNX Runtime and model deployment.", -0.3, 0.01),
            Segment(9.8, 15.0, "Thanks for listening, see you next week.", -0.25, 0.01),
        ]
        self.error = error
        self.calls = 0
        self.languages = []
        self.options = []  # DecodingOptions of each call
        self.gate: threading.Event | None = None  # set by a test to hold transcriptions until released

    def transcribe(self, audio, language=None, on_progress=None, options=None):
        self.calls += 1
        self.languages.append(language)
        self.options.append(options)
        if on_progress:
            on_progress(0.5)  # as if half the windows were done
        if self.gate is not None:
            self.gate.wait(10)
        if self.error:
            raise self.error
        return TranscriptionResult(language=language or "en", segments=list(self.segments))


def make_wav(seconds=1.0, rate=16000, channels=1, freq=440.0) -> bytes:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    samples = (0.3 * np.sin(2 * math.pi * freq * t) * 32767).astype(np.int16)
    if channels > 1:
        samples = np.repeat(samples[:, None], channels, axis=1).ravel()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        upload_dir=str(tmp_path / "uploads"),
        benchmark_dir=str(tmp_path / "benchmark"),
        models_dir=str(tmp_path / "models"),
        whisper_model_dir=str(tmp_path / "models" / "whisper-tiny"),
        default_language="auto",
        max_upload_mb=1,
        max_audio_seconds=60,
        api_token=None,  # independent of an API_TOKEN exported in the developer's shell
        vad_model_path=str(tmp_path / "no-vad.onnx"),  # tests that need VAD pass a fake
    )


@pytest.fixture
def transcriber():
    return FakeTranscriber()


@pytest.fixture
def client(settings, transcriber):
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as c:
        yield c


def wait_for(client, job_id, timeout=10.0, headers=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")
