"""The testbench's options on the service: INT4 weights (ONNX Runtime's MatMulNBits) and the same model on
CTranslate2 (faster-whisper), plus per-job memory and weight size in the metrics."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr import OnnxWhisperTranscriber, onnx_precision
from app.ct2 import CT2Engine
from app.main import Components, create_app
from app.models import describe_export
from tests.conftest import FakeTranscriber, make_wav, wait_for
from tests.test_api import upload

ROOT = Path(__file__).resolve().parents[1]


def export_module():
    spec = importlib.util.spec_from_file_location("export_whisper_onnx", ROOT / "scripts" / "export_whisper_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_graph(path: Path) -> np.ndarray:
    """x [1, 64] -> MatMul(64x64) -> Relu -> MatMul(64x32): the shape of a transformer's projections."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w1, w2 = rng.standard_normal((64, 64), dtype=np.float32) * 0.1, rng.standard_normal((64, 32), dtype=np.float32) * 0.1
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w1"], ["h"]), helper.make_node("Relu", ["h"], ["r"]),
         helper.make_node("MatMul", ["r", "w2"], ["y"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return np.maximum(rng.standard_normal((1, 64), dtype=np.float32) @ w1, 0) @ w2  # unused: shapes only


def test_int4_weights_are_a_quarter_and_run_on_onnx_runtime(tmp_path):
    pytest.importorskip("onnx_ir")
    import onnxruntime as ort

    fp32 = tmp_path / "encoder_model.onnx"
    tiny_graph(fp32)
    int4 = tmp_path / "encoder_model.int4.onnx"
    shutil.copy(fp32, int4)
    export_module().quantize_int4(int4)
    assert onnx_precision(fp32) == "fp32" and onnx_precision(int4) == "int4"
    assert int4.stat().st_size < fp32.stat().st_size / 3  # 4-bit weights and a scale per block of 32
    x = np.random.default_rng(1).standard_normal((1, 64), dtype=np.float32)
    want = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    got = ort.InferenceSession(str(int4), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    assert np.abs(got - want).max() < 0.15 * np.abs(want).max()  # close, not equal


def test_an_export_with_int4_offers_it(tmp_path):
    for name, marker in (("encoder_model.onnx", b"Conv"), ("encoder_model.int8.onnx", b"MatMulInteger"),
                         ("encoder_model.int4.onnx", b"MatMulNBits"), ("decoder_model_merged.onnx", b"MatMulInteger"),
                         ("decoder_model_merged.int4.onnx", b"MatMulNBits")):
        (tmp_path / name).write_bytes(b"\0" * 64 + marker + b"\0" * 64)
    info = describe_export(tmp_path)
    assert info.available_precisions == {"encoder": ["fp32", "int4", "int8"], "decoder": ["int4", "int8"]}
    # The weights a job ran with: the files of the precisions it used.
    t = SimpleNamespace(precision={"encoder": "fp32", "decoder": "int8"},
                        variants={"encoder": {"fp32": tmp_path / "encoder_model.onnx", "int4": tmp_path / "encoder_model.int4.onnx"},
                                  "decoder": {"int8": tmp_path / "decoder_model_merged.onnx", "int4": tmp_path / "decoder_model_merged.int4.onnx"}})
    (tmp_path / "decoder_model_merged.int4.onnx_data").write_bytes(b"\0" * 1_000_000)  # external weights count too
    assert OnnxWhisperTranscriber.weights_mb(t, {"encoder": "int4", "decoder": "int4"}) == 1.0
    assert OnnxWhisperTranscriber.weights_mb(t, {"encoder": "int8", "decoder": "int4"}) is None


class FakeWhisperModel:
    """faster_whisper.WhisperModel: records how it was loaded and asked."""

    loads, calls = [], []

    def __init__(self, path, device, compute_type, cpu_threads, local_files_only):
        FakeWhisperModel.loads.append({"path": path, "device": device, "compute_type": compute_type,
                                       "local_files_only": local_files_only})

    def transcribe(self, samples, **kw):
        FakeWhisperModel.calls.append({"samples": samples, **kw})
        segs = [SimpleNamespace(start=0.0, end=1.2, text=" Hello from CTranslate2.", avg_logprob=-0.2, no_speech_prob=0.01),
                SimpleNamespace(start=1.2, end=2.0, text=" Bye.", avg_logprob=-0.4, no_speech_prob=0.02)]
        return iter(segs), SimpleNamespace(language="en")


@pytest.fixture
def ct2_dir(tmp_path, monkeypatch):
    path = tmp_path / "ct2"
    path.mkdir()
    (path / "model.bin").write_bytes(b"\0" * 2_000_000)
    (path / "model_id.txt").write_text("openai/whisper-tiny\n")
    FakeWhisperModel.loads, FakeWhisperModel.calls = [], []
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name in ("ctranslate2", "faster_whisper")
                        else importlib.machinery.PathFinder.find_spec(name))
    return path


def test_a_job_on_ctranslate2(settings, ct2_dir, monkeypatch):
    resets = []
    monkeypatch.setattr("app.jobs.reset_peak_memory", lambda: resets.append(1) or True)
    with TestClient(create_app(settings, Components(transcriber=FakeTranscriber(), ct2=CT2Engine(ct2_dir, 2)))) as client:
        engines = client.get("/config").json()["engines"]
        assert engines["onnx"]["precisions"] == ["fp32", "int8", "int4"]
        ct2 = engines["ctranslate2"]
        assert ct2["available"] and ct2["model"] == "openai/whisper-tiny" and ct2["size_mb"] == 2.0
        assert ct2["compute_types"] == ["int8", "int8_float32", "float32"]

        res = upload(client, data={"engine": "ctranslate2", "compute_type": "float32", "beam_size": "5", "vad": "true",
                                   "vad_threshold": "0.35", "initial_prompt": "CTranslate2",
                                   "temperature_fallback": "false"})
        assert res.status_code == 202, res.text  # VAD is faster-whisper's own: the service's isn't needed
        job = wait_for(client, res.json()["id"])
        assert job["status"] == "completed", job["error"]
        assert job["engine"] == "ctranslate2" and job["precision"] == {"encoder": "float32", "decoder": "float32"}
        assert job["metrics"]["engine"] == "ctranslate2" and job["metrics"]["model_mb"] == 2.0 and resets
        transcript = client.get(f"/jobs/{job['id']}/transcript").json()
        assert transcript["text"] == "Hello from CTranslate2. Bye." and transcript["segments"][1]["start"] == 1.2
        [load] = FakeWhisperModel.loads
        assert load == {"path": str(ct2_dir), "device": "cpu", "compute_type": "float32", "local_files_only": True}
        [call] = FakeWhisperModel.calls
        assert call["samples"].dtype == np.float32 and call["beam_size"] == 5 and call["language"] == "en"
        assert call["vad_filter"] and call["vad_parameters"] == {"threshold": 0.35, "min_silence_duration_ms": 500}
        assert call["temperature"] == 0.0 and call["initial_prompt"] == "CTranslate2"

        # ONNX jobs say so too, with the size of the weights they ran with when it's known (not for this fake).
        onnx = wait_for(client, upload(client).json()["id"])
        assert onnx["engine"] == "onnx" and onnx["metrics"]["engine"] == "onnx" and onnx["metrics"]["model_mb"] is None


def test_what_ctranslate2_refuses(settings, ct2_dir, tmp_path):
    with TestClient(create_app(settings, Components(transcriber=FakeTranscriber(), ct2=CT2Engine(ct2_dir)))) as client:
        bad = upload(client, data={"engine": "ctranslate2", "encoder": "int8"})
        assert bad.status_code == 422 and "compute_type" in bad.json()["detail"]
        assert upload(client, data={"engine": "ctranslate2", "model": "openai/whisper-small"}).status_code == 422
        assert upload(client, data={"compute_type": "int8"}).status_code == 422  # ONNX takes encoder/decoder
        assert upload(client, data={"engine": "tensorrt"}).status_code == 422
        assert upload(client, data={"encoder": "int4"}).status_code == 422  # this fake has no int4 export
    with TestClient(create_app(settings, Components(transcriber=FakeTranscriber(), ct2=CT2Engine(tmp_path / "none")))) as client:
        res = upload(client, data={"engine": "ctranslate2"})
        assert res.status_code == 503 and "export with scripts/export_whisper_onnx.py --ct2" in res.json()["detail"]
        assert client.get("/config").json()["engines"]["ctranslate2"]["available"] is False
    with TestClient(create_app(settings, Components(transcriber=FakeTranscriber(), ct2=CT2Engine(ct2_dir),
                                                    memory_limit_mb=10))) as client:
        res = upload(client, data={"engine": "ctranslate2"})
        assert res.status_code == 422 and "CTranslate2 needs about" in res.json()["detail"]


_ = make_wav
