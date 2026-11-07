"""Choosing the speech model per job: the service's own, or another export in MODELS_DIR loaded for the job."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr import Segment
from app.main import Components, create_app
from app.models import ModelSet, describe_export
from tests.conftest import FakeTranscriber, make_wav, wait_for


def export(models_dir: Path, folder: str, model_id: str, size: int = 1000, multilingual: bool = False) -> Path:
    path = models_dir / folder
    path.mkdir(parents=True)
    with (path / "encoder_model.onnx").open("wb") as f:
        f.truncate(size)  # sparse: the size counts, no disk is used
    (path / "decoder_model_merged.onnx").write_bytes(b"MatMulInteger")  # an int8 graph
    (path / "tokenizer.json").write_text("{}")
    (path / "model_id.txt").write_text(model_id + "\n")
    config = {"lang_to_id": {"<|en|>": 1, "<|it|>": 2}} if multilingual else {"is_multilingual": False}
    (path / "generation_config.json").write_text(json.dumps(config))
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
    export(root, "whisper", "openai/whisper-tiny.en")  # the service's own (WHISPER_MODEL_DIR)
    export(root, "whisper-base.en", "openai/whisper-base.en")
    export(root, "whisper-small", "openai/whisper-small", multilingual=True)  # skipped: not English-only
    export(root, "whisper-small.en", "openai/whisper-small.en", size=400_000_000)
    settings.whisper_model_dir = str(root / "whisper")
    return root


def make_client(settings, transcriber, loader, limit_mb=None):
    transcriber.name = "openai/whisper-tiny.en"
    return TestClient(create_app(settings, Components(transcriber=transcriber, load_model=loader,
                                                      memory_limit_mb=limit_mb)))


def test_describe_export(models):
    info = describe_export(models / "whisper-small.en")
    assert info.id == "openai/whisper-small.en" and info.precision == {"encoder": "fp32", "decoder": "int8"}
    assert info.available_precisions == {"encoder": ["fp32"], "decoder": ["int8"]}


def test_a_job_can_use_another_model(settings, transcriber, models):
    loader = Loader()
    with make_client(settings, transcriber, loader) as client:
        config = client.get("/config").json()
        # The multilingual export (whisper-small) is left out.
        assert [(m["id"], m["default"]) for m in config["models"]] == [
            ("openai/whisper-tiny.en", True), ("openai/whisper-base.en", False), ("openai/whisper-small.en", False)]
        small = config["models"][2]
        assert small["needs_mb"] > 400 and small["fits"] is None  # no limit known
        assert "multilingual" not in small and "languages" not in small

        job = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")},
                          data={"model": "whisper-small.en"}).json()
        done = wait_for(client, job["id"])
        assert done["status"] == "completed" and done["model"] == "openai/whisper-small.en"
        assert done["requested_model"] == "openai/whisper-small.en"
        text = client.get(f"/jobs/{job['id']}/transcript").json()
        assert text["segments"][0]["text"] == "Transcribed by whisper-small.en."
        assert text["model"] == "openai/whisper-small.en" and text["language"] == "en"
        assert loader.loaded == ["openai/whisper-small.en"] and transcriber.calls == 0

        # Without a model: the service's own, nothing loaded.
        job = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")}).json()
        assert wait_for(client, job["id"])["model"] == "openai/whisper-tiny.en"
        assert loader.loaded == ["openai/whisper-small.en"]


def test_model_requests_are_checked_before_queuing(settings, transcriber, models):
    with make_client(settings, transcriber, Loader(), limit_mb=512) as client:
        wav = {"file": ("a.wav", make_wav(1.0), "audio/wav")}
        res = client.post("/jobs", files=wav, data={"model": "whisper-large"})
        assert res.status_code == 422 and "Unknown model" in res.json()["detail"]
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-small"})  # multilingual: not offered
        assert res.status_code == 422 and "Unknown model" in res.json()["detail"]
        # 512 MB can't hold whisper-small.en next to the service's model: refused with the reason, not killed.
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-small.en"})
        assert res.status_code == 422 and "larger plan" in res.json()["detail"]
        small = next(m for m in client.get("/config").json()["models"] if m["id"] == "openai/whisper-small.en")
        assert small["fits"] is False and "Render Standard" in small["reason"]
        res = client.post("/jobs", files=wav, data={"model": "openai/whisper-base.en", "decoder": "fp32"})
        assert res.status_code == 422 and "isn't available for openai/whisper-base.en" in res.json()["detail"]
        assert client.post("/jobs", files=wav, data={"model": "openai/whisper-base.en"}).status_code == 202


def test_models_fit_when_the_memory_allows():
    small = describe_export.__globals__["ModelInfo"](id="s", path=Path("."), size_mb=500)
    models = ModelSet(None, None, None, None, limit_mb=2048, rss_mb=lambda: 300)
    assert models.fits(small) == (True, None)
    models.limit_mb = 512
    assert models.fits(small)[0] is False
