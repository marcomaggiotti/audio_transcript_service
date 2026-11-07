"""The training worker (app/training_worker.py) on a tiny random Whisper: each part, a whole run and its ONNX export,
loaded back by the service. Needs PyTorch and transformers (the training Python); skipped without them."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from app import training_worker as worker  # noqa: E402
from tests.conftest import make_wav  # noqa: E402
from tests.tiny_whisper import make_tiny  # noqa: E402

ROOT = Path(__file__).parents[1]
TEXTS = ["The tin heron.", "Hello there, Aipollo!", "In the end the tin won.", "A heron on the tin.", "Then the end."]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A base model as the service keeps one: its export folder, with the PyTorch weights in torch/."""
    folder = tmp_path_factory.mktemp("base")
    make_tiny(folder / "torch")
    (folder / "model_id.txt").write_text("tiny/whisper-test.en\n")
    return {"id": "tiny/whisper-test.en", "dir": str(folder)}


@pytest.fixture
def clips(tmp_path):
    out = []
    for i, text in enumerate(TEXTS):
        path = tmp_path / f"clip{i}.wav"
        path.write_bytes(make_wav(1.0 + i * 0.3, freq=300 + 80 * i))
        out.append({"id": f"{i:032x}", "name": f"Clip {i}", "audio": str(path), "reference": text})
    return out


def test_the_tokenizer_writes_the_labels(base):
    out = worker.inspect_tokenizer({"base": base, "tokenizer": {}, "text": "Hello  there, tin!"})
    assert [t["token"] for t in out["prefix"]] == ["<|startoftranscript|>", "<|notimestamps|>"]  # English-only
    assert out["text"] == "Hello there, tin!" and out["round_trip"] and out["decoded"] == "Hello there, tin!"
    assert out["decoded_with_special"].startswith("<|startoftranscript|><|notimestamps|>Hello")
    assert out["tokens"][-1]["token"] == "<|endoftext|>" and out["tokens"][0]["special"]
    assert out["attributes"]["language"] is None and out["attributes"]["task"] is None
    lower = worker.inspect_tokenizer({"base": base, "tokenizer": {"text": "normalized"}, "text": "Hello there, Tin!"})
    assert lower["text"] == "hello there tin"
    timestamps = worker.inspect_tokenizer({"base": base, "tokenizer": {"predict_timestamps": True}, "text": "x"})
    assert [t["token"] for t in timestamps["prefix"]] == ["<|startoftranscript|>"]


def test_the_processor_and_the_collator(base, clips):
    out = worker.inspect_processor({"base": base, "clip": clips[1], "processor": {}, "tokenizer": {}})
    assert out["input_features_shape"] == [80, 3000] and out["attributes"]["sampling_rate"] == 16000
    assert out["clip"]["seconds"] == 1.3 and out["frames_with_audio"] == 130 and len(out["mel"]) == 80
    assert len(out["mel"][0]) == 130 and out["labels"] == len(out["label_tokens"])

    batch = worker.inspect_collator({"base": base, "clips": clips[:3], "collator": {}, "tokenizer": {}})
    assert batch["input_features_shape"] == [3, 80, 3000] and batch["start_token_cut"]
    rows = batch["rows"]
    assert rows[0]["tokens"][0]["token"] == "<|notimestamps|>"  # <|startoftranscript|> cut: the model adds it
    assert batch["labels_shape"][1] == max(r["length"] for r in rows) and any(r["padding"] for r in rows)
    assert all(t["id"] == -100 and t["token"] is None for r in rows for t in r["tokens"][r["length"]:])

    kept = worker.inspect_collator({"base": base, "clips": clips[:2], "tokenizer": {},
                                    "collator": {"strip_start_token": False, "pad_to_multiple_of": 8}})
    assert not kept["start_token_cut"] and kept["labels_shape"][1] % 8 == 0
    assert kept["rows"][0]["tokens"][0]["token"] == "<|startoftranscript|>"


def test_the_command_line_inspects_through_stdin(base):
    def run(request):
        proc = subprocess.run([sys.executable, "-m", "app.training_worker", "inspect"], input=json.dumps(request),
                              capture_output=True, text=True, cwd=ROOT, timeout=300, check=False)
        return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])
    code, out = run({"kind": "tokenizer", "base": base, "text": "the tin"})
    assert code == 0 and out["decoded"] == "the tin"
    code, out = run({"kind": "processor", "base": base, "clip": {"audio": "/nowhere.wav"}})
    assert code == 1 and out["error"].startswith("FileNotFoundError")


def test_unusable_clips_are_skipped_with_the_reason(tmp_path, base, clips):
    long = tmp_path / "long.wav"
    long.write_bytes(make_wav(31.0))
    extra = [{"id": "f" * 32, "name": "Too long", "audio": str(long), "reference": "long"},
             {"id": "e" * 32, "name": "Empty", "audio": clips[0]["audio"], "reference": " "},
             {"id": "d" * 32, "name": "Gone", "audio": str(tmp_path / "gone.wav"), "reference": "gone"}]
    processor = worker.load_processor(base, {})
    examples, skipped = worker.prepare(clips[:1] + extra, processor, {"tokenizer": {}, "processor": {}})
    assert [e["id"] for e in examples] == [clips[0]["id"]] and examples[0]["input_features"].shape == (80, 3000)
    reasons = {s["name"]: s["reason"] for s in skipped}
    assert "longer than 30 s" in reasons["Too long"] and reasons["Empty"] == "no correct text"
    assert reasons["Gone"].startswith("audio unreadable")
    too_many = worker.prepare(clips[:1], processor, {"tokenizer": {"max_label_tokens": 5}, "processor": {}})[1]
    assert "more than 5" in too_many[0]["reason"]


def test_the_split_is_seeded_and_keeps_both_sides(clips):
    train, test = worker.split(clips, 0.4, seed=1)
    assert len(train) == 3 and len(test) == 2 and worker.split(clips, 0.4, seed=1) == (train, test)
    assert len(worker.split(clips[:2], 0.5, seed=1)[0]) == 1 and len(worker.split(clips[:2], 0.1, seed=1)[1]) == 1


def run_request(tmp_path, base, clips, export=True, **args):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = {"base": base, "clips": clips, "data": {"test_size": 0.4, "seed": 1}, "tokenizer": {}, "processor": {},
               "collator": {}, "metric": {"name": "wer"}, "threads": 1, "work_dir": str(tmp_path / "work"),
               "args": {"learning_rate": 1e-3, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                        "warmup_steps": 0, "max_steps": 2, "eval_steps": 1, "logging_steps": 1,
                        "per_device_eval_batch_size": 2, "generation_max_length": 16, "seed": 3, **args},
               "export": {"enabled": export, "model_id": "finetuned/tiny-test",
                          "output": str(tmp_path / "models" / "tiny-test"),
                          "script": str(ROOT / "scripts" / "export_whisper_onnx.py")}}
    (run_dir / "request.json").write_text(json.dumps(request))
    return run_dir


def test_a_run_trains_scores_and_exports(tmp_path, base, clips):
    run_dir = run_request(tmp_path, base, clips)
    assert worker.train(run_dir) == 0
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phase"] == "done" and status["error"] is None
    assert (status["data"]["train"], status["data"]["test"], status["data"]["skipped"]) == (3, 2, [])
    assert [h["step"] for h in status["history"] if "loss" in h] == [1, 2]
    assert [h["step"] for h in status["history"] if "eval_wer" in h] == [1, 2]
    for part in ("baseline", "final"):
        assert set(status[part]) >= {"wer", "cer", "loss", "examples"} and len(status[part]["examples"]) == 2
    assert status["model_id"] == "finetuned/tiny-test" and status["best_step"] in (1, 2) and status["device"] == "cpu"

    model_dir = tmp_path / "models" / "tiny-test"
    assert (model_dir / "model_id.txt").read_text().strip() == "finetuned/tiny-test"
    assert (model_dir / "encoder_model.onnx").exists() and (model_dir / "torch" / "model.safetensors").exists()
    assert not (tmp_path / "work").exists() and not (run_dir / "model").exists()  # moved into the export

    from app.asr import OnnxWhisperTranscriber  # the service's own runtime loads the export
    transcriber = OnnxWhisperTranscriber.from_dir(str(model_dir), max_new_tokens=4)
    assert transcriber.name == "finetuned/tiny-test"
    transcriber.transcribe(np.zeros(16000, np.float32))  # it runs (a random model writes noise)
    # and it can be trained further: its PyTorch weights are next to the ONNX files
    assert worker.model_source({"id": "finetuned/tiny-test", "dir": str(model_dir)}) == (str(model_dir / "torch"),) * 2


def test_a_stop_ends_training_and_keeps_what_it_learned(tmp_path, base, clips):
    run_dir = run_request(tmp_path, base, clips, export=False, max_steps=50, eval_steps=50)
    (run_dir / "stop").write_text("")
    assert worker.train(run_dir) == 0
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phase"] == "done" and status["stopped_at"] == 1 and status["step"] == 1 and "final" in status
    assert (run_dir / "model" / "model.safetensors").exists() and status["model_dir"] == str(run_dir / "model")


def test_a_failure_is_reported(tmp_path, base, clips):
    run_dir = run_request(tmp_path, base, clips[:1])  # one clip: nothing left to train on
    assert worker.train(run_dir) == 1
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phase"] == "failed" and "Not enough usable clips" in status["error"]


def test_the_api_drives_the_real_worker(settings, transcriber):
    """The whole path: the service's API, its manager, this Python as the training Python, the export."""
    from fastapi.testclient import TestClient

    from app.main import Components, create_app
    from tests.test_training import add_clips, wait_run

    make_tiny(Path(settings.whisper_model_dir) / "torch")  # the service's model, with its PyTorch weights
    with TestClient(create_app(settings, Components(transcriber=transcriber, training_python=sys.executable))) as c:
        info = c.get("/training").json()
        assert info["available"] and info["device"] == "cpu" and info["torch"].startswith("2.")
        clips = add_clips(c, 4)
        assert c.post("/training/inspect/tokenizer", json={"text": "Hello there"}).json()["decoded"] == "Hello there"
        assert c.post("/training/inspect/processor", json={"clip_id": clips[0]}).json()["input_features_shape"] == [80, 3000]
        batch = c.post("/training/inspect/collator", json={"clip_ids": clips[:2]}).json()
        assert batch["input_features_shape"] == [2, 80, 3000] and batch["start_token_cut"]
        run = c.post("/training/runs", json={"name": "tiny", "data": {"test_size": 0.25}, "args": {
            "max_steps": 2, "eval_steps": 1, "logging_steps": 1, "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2, "generation_max_length": 16, "learning_rate": 1e-3, "warmup_steps": 0}}).json()
        run = wait_run(c, run["id"], timeout=600)
        assert run["status"] == "completed" and run["exported"] and run["data"]["test"] == 1, run
        assert run["model_id"] in {m["id"] for m in c.get("/config").json()["models"]}
        assert run["final"]["examples"] and run["history"]
