import io
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr import DecodingOptions, Segment
from app.benchmark import clips_from_dir
from app.main import Components, create_app
from tests.conftest import FakeTranscriber, make_wav, wait_for

FULL_TEXT = ("Welcome to the show about machine learning. Today we talk about ONNX Runtime and model deployment. "
             "Thanks for listening, see you next week.")


def other_model(path: Path) -> FakeTranscriber:
    model = FakeTranscriber(segments=[Segment(0.0, 5.0, "Welcome to the shoe about machine leaning.")])
    model.name = (path / "model_id.txt").read_text().strip()
    return model


@pytest.fixture
def models_dir(settings):
    base = Path(settings.models_dir) / "whisper-base"
    base.mkdir(parents=True)
    for name in ("encoder_model.onnx", "tokenizer.json"):
        (base / name).write_bytes(b"")
    (base / "model_id.txt").write_text("openai/whisper-base\n")
    (Path(settings.models_dir) / "not-a-model").mkdir()
    return Path(settings.models_dir)


@pytest.fixture
def bench(settings, transcriber, models_dir):
    with TestClient(create_app(settings, Components(transcriber=transcriber, load_model=other_model))) as c:
        yield c


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def add_source(client, seconds=20.0):
    response = client.post("/benchmark/sources", files={"file": ("episode.wav", make_wav(seconds), "audio/wav")},
                           data={"language": "en"})
    assert response.status_code == 202, response.text
    source = response.json()
    wait_for(client, source["job_id"])
    return client.get(f"/benchmark/sources/{source['id']}").json()


def wait_run(client, run_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/benchmarks/{run_id}").json()
        if run["status"] in ("completed", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish: {run}")


def test_recording_proposes_intervals(bench):
    source = add_source(bench)
    assert source["status"] == "completed" and source["duration_seconds"] == 20.0
    assert [(s["start"], s["end"]) for s in source["segments"]] == [(0.0, 4.2), (4.2, 9.8), (9.8, 15.0)]
    assert source["media_type"] == "audio/wav"
    assert bench.get(f"/benchmark/sources/{source['id']}/audio").content[:4] == b"RIFF"
    listed = bench.get("/benchmark/sources").json()
    assert [s["id"] for s in listed] == [source["id"]] and "segments" not in listed[0]


def test_interval_becomes_a_reference_clip(bench):
    source = add_source(bench)
    response = bench.post("/benchmark/clips", json={
        "source_id": source["id"], "start": 4.2, "end": 9.8,
        "reference": " Today we talk about ONNX Runtime and model deployment. ",
    })
    assert response.status_code == 201, response.text
    clip = response.json()
    assert clip["duration_seconds"] == 5.6 and (clip["start"], clip["end"]) == (4.2, 9.8)
    assert clip["reference"] == "Today we talk about ONNX Runtime and model deployment."
    assert clip["name"] == "episode.wav 0:04.2" and clip["source_filename"] == "episode.wav"
    assert abs(wav_seconds(bench.get(f"/benchmark/clips/{clip['id']}/audio").content) - 5.6) < 0.01

    edited = bench.patch(f"/benchmark/clips/{clip['id']}", json={"reference": "Today we talk.", "name": "intro"})
    assert edited.json()["reference"] == "Today we talk." and edited.json()["name"] == "intro"
    assert [c["reference"] for c in bench.get("/benchmark/clips").json()] == ["Today we talk."]

    # Clips keep their own audio when the recording is deleted.
    assert bench.delete(f"/benchmark/sources/{source['id']}").status_code == 204
    assert bench.get(f"/benchmark/clips/{clip['id']}/audio").status_code == 200
    assert bench.delete(f"/benchmark/clips/{clip['id']}").status_code == 204
    assert bench.get("/benchmark/clips").json() == []


@pytest.mark.parametrize("start,end,message", [
    (5.0, 5.0, "end after it starts"),
    (5.0, 5.2, "at least"),
    (25.0, 30.0, "after the end of the recording"),
    (0.0, 400.0, "at most"),
])
def test_bad_intervals(bench, start, end, message):
    source = add_source(bench)
    response = bench.post("/benchmark/clips", json={"source_id": source["id"], "start": start, "end": end,
                                                     "reference": "x"})
    assert response.status_code == 422 and message in response.json()["detail"]


def test_upload_a_ready_clip(bench):
    response = bench.post("/benchmark/clips/upload", files={"file": ("clip01.wav", make_wav(3.0), "audio/wav")},
                          data={"reference": "Hello there."})
    assert response.status_code == 201
    assert response.json()["name"] == "clip01" and response.json()["duration_seconds"] == 3.0
    assert response.json()["source_id"] is None


def test_models_and_variants(bench):
    models = bench.get("/benchmark/models").json()
    assert models == [
        {"id": "whisper-tiny", "name": "fake-whisper", "default": True},
        {"id": "whisper-base", "name": "openai/whisper-base", "default": False},
    ]
    assert [v["id"] for v in bench.get("/benchmark/variants").json()] == ["greedy", "fallback", "fallback_previous"]


def test_benchmark_run_scores_models_and_variants(bench, transcriber, settings):
    source = add_source(bench)
    for start, end in ((0.0, 4.2), (4.2, 9.8)):
        bench.post("/benchmark/clips", json={"source_id": source["id"], "start": start, "end": end,
                                             "reference": FULL_TEXT})
    transcriber.options.clear()

    response = bench.post("/benchmarks", json={"models": ["whisper-tiny", "whisper-base"],
                                               "variants": ["greedy", "fallback_previous"], "language": "en",
                                               "initial_prompt": "ONNX"})
    assert response.status_code == 202, response.text
    run = wait_run(bench, response.json()["id"])
    assert run["status"] == "completed"
    assert run["progress"]["done"] == run["progress"]["total"] == 8
    by_key = {(r["model"], r["variant"]): r for r in run["results"]}
    assert list(by_key) == [("whisper-tiny", "greedy"), ("whisper-tiny", "fallback_previous"),
                            ("whisper-base", "greedy"), ("whisper-base", "fallback_previous")]

    tiny = by_key["whisper-tiny", "greedy"]
    assert tiny["wer"] == 0.0 and tiny["reference_words"] == 2 * 23 and len(tiny["clips"]) == 2
    assert tiny["clips"][0]["alignment"][0] == ["ok", "welcome", "welcome"]
    assert tiny["real_time_factor"] is not None and tiny["peak_memory_mb"] > 0

    base = by_key["whisper-base", "greedy"]
    assert base["model_name"] == "openai/whisper-base" and base["load_seconds"] is not None
    # 2 substitutions (shoe, leaning) and 16 deletions per clip, out of 23 reference words.
    assert (base["substitutions"], base["deletions"], base["insertions"]) == (4, 32, 0)
    assert base["wer"] == round(36 / 46, 4)

    assert transcriber.options == [
        DecodingOptions(initial_prompt="ONNX")] * 2 + [
        DecodingOptions(temperature_fallback=True, condition_on_previous_text=True, initial_prompt="ONNX")] * 2
    assert transcriber.languages[-4:] == ["en"] * 4

    listed = bench.get("/benchmarks").json()
    assert [r["id"] for r in listed] == [run["id"]] and "clips" not in listed[0]["results"][0]

    # Finished runs are kept on disk, and the clips folder is a plain audio + .txt dataset.
    with TestClient(create_app(settings, Components(transcriber=transcriber, load_model=other_model))) as again:
        assert again.get(f"/benchmarks/{run['id']}").json()["results"] == run["results"]
    clips = clips_from_dir(Path(settings.benchmark_dir) / "clips")
    assert [c.reference for c in clips] == [FULL_TEXT, FULL_TEXT]
    assert bench.delete(f"/benchmarks/{run['id']}").status_code == 204


def test_model_that_fails_to_load_is_reported(settings, transcriber, models_dir):
    def broken(path):
        raise RuntimeError("bad export")

    with TestClient(create_app(settings, Components(transcriber=transcriber, load_model=broken))) as client:
        client.post("/benchmark/clips/upload", files={"file": ("a.wav", make_wav(2.0), "audio/wav")},
                    data={"reference": FULL_TEXT})
        run = client.post("/benchmarks", json={"models": ["whisper-base", "whisper-tiny"],
                                               "variants": ["greedy"]}).json()
        run = wait_run(client, run["id"])
        assert run["status"] == "completed"
        assert run["results"][0]["error"] == "Model didn't load: bad export" and run["results"][0]["wer"] is None
        assert run["results"][1]["wer"] == 0.0


def test_benchmark_validation(bench):
    response = bench.post("/benchmarks", json={"models": ["whisper-tiny"], "variants": ["greedy"]})
    assert response.status_code == 422 and "No reference clips" in response.json()["detail"]
    bench.post("/benchmark/clips/upload", files={"file": ("a.wav", make_wav(2.0), "audio/wav")},
               data={"reference": "x"})
    response = bench.post("/benchmarks", json={"models": ["whisper-huge"], "variants": ["greedy"]})
    assert response.status_code == 422 and "whisper-huge" in response.json()["detail"]
    response = bench.post("/benchmarks", json={"models": ["whisper-tiny"], "variants": ["beam"]})
    assert response.status_code == 422 and "beam" in response.json()["detail"]
    response = bench.post("/benchmarks", json={"models": ["whisper-tiny"], "variants": ["greedy"],
                                               "clip_ids": ["f" * 32]})
    assert response.status_code == 422 and "Unknown clip" in response.json()["detail"]


def test_ids_cannot_reach_outside_the_store(bench):
    for path in ("/benchmark/sources/..%2F..%2Fsecret", "/benchmark/clips/x/audio", "/benchmarks/" + "g" * 32):
        assert bench.get(path).status_code == 404


def test_interrupted_run_is_marked_failed(settings, transcriber):
    from app.benchmark import BenchmarkStore
    from app.schemas import BenchmarkRun

    store = BenchmarkStore(settings.benchmark_dir)
    store.setup()
    store.save_run(BenchmarkRun(id="a" * 32, status="processing", models=["m"], variants=["greedy"],
                                language="en", clip_ids=[]))
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as client:
        run = client.get(f"/benchmarks/{'a' * 32}").json()
        assert run["status"] == "failed" and "restart" in run["error"]


def test_decoding_options_on_jobs(client, transcriber):
    client.post("/jobs", files={"file": ("a.wav", make_wav(), "audio/wav")})
    job = client.post("/jobs", files={"file": ("a.wav", make_wav(), "audio/wav")},
                      data={"temperature_fallback": "false", "initial_prompt": " ONNX, Whisper "}).json()
    status = wait_for(client, job["id"])
    assert (status["temperature_fallback"], status["condition_on_previous_text"]) == (False, True)
    assert transcriber.options == [
        DecodingOptions(temperature_fallback=True, condition_on_previous_text=True),
        DecodingOptions(temperature_fallback=False, condition_on_previous_text=True, initial_prompt="ONNX, Whisper"),
    ]


def test_benchmark_needs_the_token(settings, transcriber):
    settings.api_token = "secret"
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as client:
        assert client.get("/benchmark/models").status_code == 401
        assert client.get("/benchmarks").status_code == 401
        assert client.get("/benchmark/models", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/").status_code == 200  # the page itself asks for the token
