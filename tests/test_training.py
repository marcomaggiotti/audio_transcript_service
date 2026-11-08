"""The training API (app/training.py, app/training_api.py) with a stand-in for the training Python: availability,
the parts' previews, runs (queued, followed, stopped, deleted, cut off), the fine-tuned model joining the models.
No PyTorch needed; tests/test_training_worker.py runs the real worker."""

import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app.main import Components, create_app
from app.training_worker import error_rates
from tests.conftest import make_wav

# Answers like the training Python: the availability probe (-c), a preview (inspect) and a run (train). A run holds
# while FAKE_HOLD names an existing file (until the test removes it, or a stop arrives); FAKE_FAIL fails it,
# FAKE_CRASH ends the process as the system's out-of-memory killer would.
FAKE = '''#!PYTHON
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ["-c"]:
    print(json.dumps({"torch": "2.14.1", "transformers": "4.57.6", "cuda": False, "device": "cpu", "cpus": 4}))
    sys.exit(0)
assert args[:2] == ["-m", "app.training_worker"], args
if args[2] == "inspect":
    request = json.loads(sys.stdin.read())
    if request.get("text") == "boom":
        print(json.dumps({"error": "ValueError: boom"}))
        sys.exit(1)
    print(json.dumps({"kind": request["kind"], "request": request}))
    sys.exit(0)
run_dir = Path(args[3])
request = json.loads((run_dir / "request.json").read_text())
def status(**kw):
    (run_dir / "status.json").write_text(json.dumps(kw))
if os.environ.get("FAKE_CRASH"):
    os._exit(137)
steps = request["args"]["max_steps"]
history = [{"step": 1, "loss": 2.5, "learning_rate": 1e-05}, {"step": 2, "eval_wer": 0.3, "eval_loss": 1.2}]
base = {"phase": "training", "step": 1, "max_steps": steps, "history": history[:1], "device": "cpu", "error": None,
        "data": {"train": len(request["clips"]) - 1, "test": 1, "skipped": []}, "baseline": {"wer": 0.5, "examples": []}}
status(**base)
hold = os.environ.get("FAKE_HOLD")
while hold and Path(hold).exists() and not (run_dir / "stop").exists():
    time.sleep(0.02)
if os.environ.get("FAKE_FAIL"):
    status(**{**base, "phase": "failed", "error": "RuntimeError: no clip could be read"})
    sys.exit(1)
stopped = (run_dir / "stop").exists()
done = {**base, "phase": "done", "step": 1 if stopped else steps, "history": history, "final": {"wer": 0.2, "examples": []},
        "best_step": 2, "stopped_at": 1 if stopped else None}
export = request["export"]
if export["enabled"]:
    out = Path(export["output"])
    out.mkdir(parents=True)
    (out / "encoder_model.onnx").write_text("not a graph")
    (out / "tokenizer.json").write_text("{}")
    (out / "model_id.txt").write_text(export["model_id"] + "\\n")
    done.update(model_id=export["model_id"], model_dir=str(out))
status(**done)
'''


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / "fake-python"
    path.write_text(FAKE.replace("PYTHON", sys.executable, 1))
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def trainer(settings, transcriber, fake_python):
    with TestClient(create_app(settings, Components(transcriber=transcriber, training_python=fake_python))) as c:
        yield c


def add_clips(client, n):
    return [client.post("/benchmark/clips/upload", files={"file": (f"c{i}.wav", make_wav(1.0 + i), "audio/wav")},
                        data={"reference": f"Clip number {i} about Aipollo.", "name": f"Clip {i}"}).json()["id"]
            for i in range(n)]


def wait_run(client, run_id, until=("completed", "failed", "stopped"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/training/runs/{run_id}").json()
        if run["status"] in until:
            return run
        time.sleep(0.03)
    raise AssertionError(run)


def test_without_pytorch_it_says_how_to_get_it(client, tmp_path):
    no_torch = tmp_path / "python-without-torch"  # like the runtime's Python: no torch
    no_torch.write_text(f"#!{sys.executable}\nraise ModuleNotFoundError(\"No module named 'torch'\")\n")
    no_torch.chmod(0o755)
    client.app.state.training.python = str(no_torch)
    info = client.get("/training").json()
    assert info["available"] is False and "build the image with --build-arg TRAINING=true" in info["error"]
    assert "No module named 'torch'" in info["error"]
    assert info["base_models"] == [{"id": "fake-whisper", "default": True, "finetuned": False}]
    assert info["defaults"]["args"]["learning_rate"] == 1e-05 and info["defaults"]["tokenizer"]["text"] == "as_written"
    res = client.post("/training/runs", json={})
    assert res.status_code == 503 and "TRAINING=true" in res.json()["detail"]
    assert client.post("/training/inspect/tokenizer", json={"text": "hi"}).status_code == 503
    client.app.state.training.python = "/nowhere/python"
    assert "not found" in client.get("/training?refresh=true").json()["error"]


def test_a_run_trains_and_its_model_joins_the_others(trainer):
    clips = add_clips(trainer, 3)
    info = trainer.get("/training").json()
    assert info["available"] and info["device"] == "cpu" and info["clips"] == 3 and info["clip_seconds"] == 6.0
    assert [c["id"] for c in info["clip_list"]] == clips and info["clip_list"][0]["reference"] == "Clip number 0 about Aipollo."
    res = trainer.post("/training/runs", json={"name": "Aipollo words!", "args": {"max_steps": 30, "eval_steps": 10},
                                               "data": {"test_size": 0.34, "seed": 7}})
    assert res.status_code == 202, res.text
    run = res.json()
    assert run["status"] in ("queued", "running") and run["name"].startswith("aipollo-words-")
    assert run["model_id"] == f"finetuned/{run['name']}" and run["base_model"] == "fake-whisper" and run["clips"] == 3

    run = wait_run(trainer, run["id"])
    assert run["status"] == "completed" and run["exported"] and run["error"] is None, run
    assert run["phase"] == "done" and run["step"] == 30 and run["best_step"] == 2
    assert [h["step"] for h in run["history"]] == [1, 2] and run["baseline"]["wer"] == 0.5 and run["final"]["wer"] == 0.2

    sent = json.loads((trainer.app.state.training.store.run_dir(run["id"]) / "request.json").read_text())
    assert sorted(c["id"] for c in sent["clips"]) == sorted(clips) and sent["data"] == {"test_size": 0.34, "seed": 7}
    assert sent["base"]["id"] == "fake-whisper" and sent["args"]["max_steps"] == 30 and sent["export"]["enabled"]
    assert sent["clips"][0]["audio"].endswith(".wav") and sent["clips"][0]["reference"].startswith("Clip number")

    models = {m["id"] for m in trainer.get("/config").json()["models"]}
    assert run["model_id"] in models  # a job can choose it now
    assert run["name"] in {m["id"] for m in trainer.get("/benchmark/models").json()}  # and the benchmark compare it
    assert {"id": run["model_id"], "default": False, "finetuned": True} in trainer.get("/training").json()["base_models"]
    assert [r["id"] for r in trainer.get("/training/runs").json()] == [run["id"]]

    assert trainer.delete(f"/training/runs/{run['id']}").status_code == 204
    assert run["model_id"] not in {m["id"] for m in trainer.get("/config").json()["models"]}
    assert trainer.get(f"/training/runs/{run['id']}").status_code == 404
    assert not (trainer.app.state.training.store.models_dir / run["name"]).exists()


def test_runs_wait_their_turn_and_stop(trainer, tmp_path, monkeypatch):
    add_clips(trainer, 2)
    hold = tmp_path / "hold"
    hold.write_text("")
    monkeypatch.setenv("FAKE_HOLD", str(hold))
    first = trainer.post("/training/runs", json={"export": False}).json()
    second = trainer.post("/training/runs", json={}).json()
    wait_run(trainer, first["id"], until=("running",))
    assert trainer.get(f"/training/runs/{second['id']}").json()["status"] == "queued"
    stopped = trainer.post(f"/training/runs/{second['id']}/stop").json()
    assert stopped["status"] == "stopped" and stopped["stop_requested"]  # never starts

    stopping = trainer.post(f"/training/runs/{first['id']}/stop").json()
    assert stopping["status"] == "running" and stopping["message"] == "Stopping after the current step"
    run = wait_run(trainer, first["id"])
    assert run["status"] == "completed" and run["stopped_at"] == 1 and not run["exported"]  # kept what it learned
    assert wait_run(trainer, second["id"])["status"] == "stopped"


def test_a_failed_and_a_killed_run_say_why(trainer, monkeypatch):
    add_clips(trainer, 2)
    monkeypatch.setenv("FAKE_FAIL", "1")
    run = wait_run(trainer, trainer.post("/training/runs", json={}).json()["id"])
    assert run["status"] == "failed" and run["error"] == "RuntimeError: no clip could be read"
    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.setenv("FAKE_CRASH", "1")
    run = wait_run(trainer, trainer.post("/training/runs", json={}).json()["id"])
    assert run["status"] == "failed" and "code 137" in run["error"] and "out of memory" in run["error"]
    assert "see its log" in run["error"] and trainer.get(f"/training/runs/{run['id']}/log").status_code == 200


def test_a_run_needs_clips_a_known_model_and_a_gpu_for_fp16(trainer):
    assert "At least 2 clips" in trainer.post("/training/runs", json={}).json()["detail"]
    clips = add_clips(trainer, 2)
    assert trainer.post("/training/runs", json={"base_model": "openai/whisper-huge"}).status_code == 422
    assert "fp16 needs a GPU" in trainer.post("/training/runs", json={"args": {"fp16": True}}).json()["detail"]
    res = trainer.post("/training/runs", json={"data": {"clip_ids": [clips[0], "a" * 32]}})
    assert res.status_code == 422 and "No such clip" in res.json()["detail"]
    assert trainer.post("/training/runs", json={"args": {"learning_rate": 0}}).status_code == 422
    assert trainer.get("/training/runs/not-an-id").status_code == 404
    assert trainer.post(f"/training/runs/{'b' * 32}/stop").status_code == 404


def test_the_previews_go_to_the_training_python(trainer):
    clips = add_clips(trainer, 3)
    out = trainer.post("/training/inspect/tokenizer", json={"text": "Aipollo", "tokenizer": {"text": "normalized"}}).json()
    assert out["kind"] == "tokenizer" and out["request"]["text"] == "Aipollo"
    assert out["request"]["base"]["id"] == "fake-whisper" and out["request"]["tokenizer"]["text"] == "normalized"
    out = trainer.post("/training/inspect/processor", json={"clip_id": clips[1]}).json()
    assert out["request"]["clip"]["id"] == clips[1] and out["request"]["clip"]["audio"].endswith(f"{clips[1]}.wav")
    out = trainer.post("/training/inspect/collator", json={"clip_ids": clips[:2] + clips[:1],
                                                           "collator": {"pad_to_multiple_of": 8}}).json()
    assert [c["id"] for c in out["request"]["clips"]] == clips[:2] and out["request"]["collator"]["pad_to_multiple_of"] == 8
    res = trainer.post("/training/inspect/tokenizer", json={"text": "boom"})
    assert res.status_code == 422 and res.json()["detail"] == "The tokenizer failed: ValueError: boom"
    assert trainer.post("/training/inspect/processor", json={"clip_id": "c" * 32}).status_code == 422
    assert trainer.post("/training/inspect/tokenizer", json={"text": "x", "base_model": "nope"}).status_code == 422


def test_the_metric_is_the_benchmark_s(client):
    out = client.post("/training/inspect/metric", json={"metric": {"name": "cer"}, "pairs": [
        {"reference": "The tin, heron.", "prediction": "the tin heron"},
        {"reference": "Hello there", "prediction": "hello their"}]}).json()
    assert out["wer"] == 0.2 and out["metric"] == "cer" and out["value"] == out["cer"] and 0 < out["cer"] < 0.2
    assert out["pairs"][1]["alignment"] == [["ok", "hello", "hello"], ["sub", "there", "their"]]
    assert error_rates(["a b"], ["a b"]) == {"wer": 0.0, "cer": 0.0, "words": 2, "word_errors": 0}


def test_runs_cut_off_by_a_restart_are_failed(settings, transcriber, fake_python, tmp_path):
    with TestClient(create_app(settings, Components(transcriber=transcriber, training_python=fake_python))) as c:
        add_clips(c, 2)
        manager = c.app.state.training
        hold = tmp_path / "hold"
        hold.write_text("")
        manager.env = lambda base=manager.env: {**base(), "FAKE_HOLD": str(hold)}
        run_id = c.post("/training/runs", json={}).json()["id"]
        wait_run(c, run_id, until=("running",))
    with TestClient(create_app(settings, Components(transcriber=transcriber, training_python=fake_python))) as c:
        run = c.get(f"/training/runs/{run_id}").json()
        assert run["status"] == "failed" and run["error"] == "Interrupted by a restart; start it again"


def test_training_needs_the_token(settings, transcriber):
    settings.api_token = "secret"
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as c:
        assert c.get("/training").status_code == 401
        assert c.post("/training/runs", json={}).status_code == 401
        assert c.get("/training", headers={"X-API-Key": "secret"}).status_code == 200
