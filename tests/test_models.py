"""Choosing the speech model per job: the service's own, or another export in MODELS_DIR loaded for the job."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr import Segment
from app.main import Components, create_app
from app.models import ModelSet, describe_export
from tests.conftest import FakeTranscriber, make_wav, wait_for


def export(models_dir: Path, folder: str, model_id: str, size: int = 1000, english_only: bool = False) -> Path:
    path = models_dir / folder
    path.mkdir(parents=True)
    with (path / "encoder_model.onnx").open("wb") as f:
        f.truncate(size)  # sparse: the size counts, no disk is used
    (path / "decoder_model_merged.onnx").write_bytes(b"MatMulInteger")  # an int8 graph
    (path / "tokenizer.json").write_text("{}")
    (path / "model_id.txt").write_text(model_id + "\n")
    lang = {} if english_only else {"<|en|>": 1, "<|it|>": 2}
    (path / "generation_config.json").write_text(json.dumps({"lang_to_id": lang}))
    return path


class Loader:
    def __init__(self):
        self.loaded, self.released = [], 0

    def __call__(self, path: Path) -> FakeTranscriber:
        model = FakeTranscriber(segments=[Segment(0.0, 2.0, f"Transcribed by {path.name}.")])
        model.name = (path / "model_id.txt").read_text().strip()
        self.loaded.append(model.name)
        return model


@pytest.fixture
def models(settings):
    root = Path(settings.models_dir)
    export(root, "whisper", "openai/whisper-tiny")  # the service's own (WHISPER_MODEL_DIR)
    export(root, "whisper-small", "openai/whisper-small", size=400_000_000)
    export(root, "whisper-tiny.en", "openai/whisper-tiny.en", english_only=True)
    settings.whisper_model_dir = str(root / "whisper")
    return root


def make_client(settings, transcriber, loader, limit_mb=None):
    transcriber.name = "openai/whisper-tiny"
    return TestClient(create_app(settings, Components(transcriber=transcriber, load_model=loader,
                                                      memory_limit_mb=limit_mb)))


def test_describe_export(models):
    info = describe_export(models / "whisper-small")
    assert info.id == "openai/whisper-small" and info.precision == {"encoder": "fp32", "decoder": "int8"}
    assert info.available_precisions == {"encoder": ["fp32"], "decoder": ["int8"]}
    assert describe_export(models / "whisper-tiny.en").languages == ["en"]


def test_a_job_can_use_another_model(settings, transcriber, models):
    loader = Loader()
    with make_client(settings, transcriber, loader) as client:
        config = client.get("/config").json()
        assert [(m["id"], m["default"]) for m in config["models"]] == [
            ("openai/whisper-tiny", True), ("openai/whisper-small", False), ("openai/whisper-tiny.en", False)]
        small = config["models"][1]
        assert small["multilingual"] and small["needs_mb"] > 400 and small["fits"] is None  # no limit known

        job = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")},
                          data={"model": "whisper-small", "language": "it"}).json()
        done = wait_for(client, job["id"])
        assert done["status"] == "completed" and done["model"] == "openai/whisper-small"
        assert done["requested_model"] == "openai/whisper-small"
        text = client.get(f"/jobs/{job['id']}/transcript").json()
        assert text["segments"][0]["text"] == "Transcribed by whisper-small." and text["model"] == "openai/whisper-small"
        assert loader.loaded == ["openai/whisper-small"] and transcriber.calls == 0

        # Without a model: the service's own, nothing loaded.
        job = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")}).json()
        assert wait_for(client, job["id"])["model"] == "openai/whisper-tiny" and loader.loaded == ["openai/whisper-small"]


def test_model_requests_are_checked_before_queuing(settings, transcriber, models):
    with make_client(settings, transcriber, Loader(), limit_mb=512) as client:
        wav = {"file": ("a.wav", make_wav(1.0), "audio/wav")}
        res = client.post("/jobs", files=wav, data={"model": "whisper-large"})
        assert res.status_code == 422 and "Unknown model" in res.json()["detail"]
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-tiny.en", "language": "it"})
        assert res.status_code == 422 and "English-only" in res.json()["detail"]
        # 512 MB can't hold whisper-small next to the service's model: refused with the reason, not killed.
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-small"})
        assert res.status_code == 422 and "larger plan" in res.json()["detail"]
        small = next(m for m in client.get("/config").json()["models"] if m["id"] == "openai/whisper-small")
        assert small["fits"] is False and "Render Standard" in small["reason"]
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-tiny.en", "decoder": "fp32"})
        assert res.status_code == 422 and "isn't available for openai/whisper-tiny.en" in res.json()["detail"]
        assert client.post("/jobs", files=wav, data={"model": "openai/whisper-tiny.en", "language": "en"}).status_code == 202


def test_models_fit_when_the_memory_allows():
    small = describe_export.__globals__["ModelInfo"](id="s", path=Path("."), size_mb=500)
    models = ModelSet(None, None, None, None, limit_mb=2048, rss_mb=lambda: 300)
    assert models.fits(small) == (True, None)
    models.limit_mb = 512
    assert models.fits(small)[0] is False
