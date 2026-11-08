"""The export's attention fusion (scripts/export_whisper_onnx.py fuse_attention): every attention block of optimum's
Whisper export becomes ONNX Runtime's MultiHeadAttention, with the same outputs, or is left as exported.

The graphs here are small copies of the export's three patterns: self-attention on projections (the encoder),
masked self-attention on the KV cache with the NaN guard (a decoder's later calls), and cross-attention on cached
encoder keys."""

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tests.test_engines import export_module

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

HEADS, HEAD, D = 2, 4, 8


def _const(name, values, dtype=np.int64):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(values, dtype=dtype), name + "_v"))


def _split(prefix, x, perm):
    """x [batch, L, D] -> Reshape [batch, L, heads, head] -> Transpose(perm), as optimum's export writes it."""
    return [helper.make_node("Reshape", [x, "heads_shape"], [prefix + "_r"]),
            helper.make_node("Transpose", [prefix + "_r"], [prefix + "_t"], perm=perm)]


def attention_graph(path: Path, cached: bool) -> None:
    """cached=False: the encoder's pattern (keys and values from projections of x). cached=True: a decoder's later
    call: masked self-attention over Concat(past, new) with the NaN guard, then cross-attention on cached encoder keys."""
    rng = np.random.default_rng(1)
    weights = {f"w{n}": rng.standard_normal((D, D)).astype(np.float32) * 0.3 for n in ("q", "k", "v", "o", "cq")}
    nodes = [_const("heads_shape", [0, 0, HEADS, HEAD]), _const("hidden_shape", [0, 0, D]),
             _const("sq", 0.5, np.float32), _const("one", 1.0, np.float32), _const("zero", 0.0, np.float32),
             helper.make_node("MatMul", ["x", "wq"], ["q_proj"]), helper.make_node("Mul", ["q_proj", "sq"], ["q"]),
             helper.make_node("MatMul", ["x", "wk"], ["k"]), helper.make_node("MatMul", ["x", "wv"], ["v"])]
    nodes += _split("q", "q", [0, 2, 1, 3]) + [helper.make_node("Mul", ["q_t", "one"], ["q_s"])]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "seq", D])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if not cached:
        nodes += _split("k", "k", [0, 2, 3, 1]) + [helper.make_node("Mul", ["k_t", "one"], ["k_s"])]
        nodes += _split("v", "v", [0, 2, 1, 3])
        nodes += [helper.make_node("MatMul", ["q_s", "k_s"], ["scores"]),
                  helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
                  helper.make_node("MatMul", ["probs", "v_t"], ["ctx"]),
                  helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0, 2, 1, 3]),
                  helper.make_node("Reshape", ["ctx_t", "hidden_shape"], ["att"]),
                  helper.make_node("MatMul", ["att", "wo"], ["y"])]
    else:
        cache = lambda name, length: helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", HEADS, length, HEAD])  # noqa: E731
        inputs += [cache("past_key", "past"), cache("past_value", "past"), cache("enc_key", "enc"), cache("enc_value", "enc"),
                   helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["batch", 1, "seq", "total"])]
        outputs += [helper.make_tensor_value_info("present_key", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("present_value", TensorProto.FLOAT, None)]
        nodes += _split("k", "k", [0, 2, 1, 3]) + _split("v", "v", [0, 2, 1, 3])
        nodes += [helper.make_node("Concat", ["past_key", "k_t"], ["present_key"], axis=2),
                  helper.make_node("Concat", ["past_value", "v_t"], ["present_value"], axis=2),
                  helper.make_node("Transpose", ["present_key"], ["k_tt"], perm=[0, 1, 3, 2]),
                  helper.make_node("Mul", ["k_tt", "one"], ["k_s"]),
                  helper.make_node("MatMul", ["q_s", "k_s"], ["scores"]),
                  helper.make_node("Add", ["scores", "mask"], ["masked"]),
                  helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
                  helper.make_node("IsNaN", ["probs"], ["nan"]),
                  helper.make_node("Where", ["nan", "zero", "probs"], ["probs_ok"]),
                  helper.make_node("MatMul", ["probs_ok", "present_value"], ["ctx"]),
                  helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0, 2, 1, 3]),
                  helper.make_node("Reshape", ["ctx_t", "hidden_shape"], ["self_att"]),
                  # cross-attention: a new query on the encoder's cached keys and values
                  helper.make_node("MatMul", ["self_att", "wcq"], ["cq_proj"]),
                  helper.make_node("Mul", ["cq_proj", "sq"], ["cq"])]
        nodes += _split("cq", "cq", [0, 2, 1, 3]) + [
            helper.make_node("Mul", ["cq_t", "one"], ["cq_s"]),
            helper.make_node("Transpose", ["enc_key"], ["ek_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Mul", ["ek_t", "one"], ["ek_s"]),
            helper.make_node("MatMul", ["cq_s", "ek_s"], ["cscores"]),
            helper.make_node("Softmax", ["cscores"], ["cprobs"], axis=-1),
            helper.make_node("MatMul", ["cprobs", "enc_value"], ["cctx"]),
            helper.make_node("Transpose", ["cctx"], ["cctx_t"], perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["cctx_t", "hidden_shape"], ["cross_att"]),
            helper.make_node("MatMul", ["cross_att", "wo"], ["y"])]
    graph = helper.make_graph(nodes, "attention", inputs, outputs,
                              [numpy_helper.from_array(w, name) for name, w in weights.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


def feeds(cached: bool, batch: int, seq: int, past: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    feed = {"x": rng.standard_normal((batch, seq, D)).astype(np.float32)}
    if cached:
        total = past + seq
        causal = np.triu(np.full((seq, total), -np.inf, np.float32), k=past + 1)  # a new token sees the past and itself
        feed |= {"past_key": rng.standard_normal((batch, HEADS, past, HEAD)).astype(np.float32),
                 "past_value": rng.standard_normal((batch, HEADS, past, HEAD)).astype(np.float32),
                 "enc_key": rng.standard_normal((batch, HEADS, 30, HEAD)).astype(np.float32),
                 "enc_value": rng.standard_normal((batch, HEADS, 30, HEAD)).astype(np.float32),
                 "mask": np.broadcast_to(causal, (batch, 1, seq, total)).copy()}
    return feed


def run(path: Path, feed: dict) -> list:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)


@pytest.mark.parametrize("cached", [False, True])
def test_each_attention_block_becomes_multihead_attention_with_the_same_outputs(tmp_path, cached):
    ex = export_module()
    exported, fused = tmp_path / "exported.onnx", tmp_path / "fused.onnx"
    attention_graph(exported, cached)
    shutil.copy(exported, fused)
    assert ex.fuse_attention(fused, HEADS) == (2 if cached else 1)
    ops = Counter(n.op_type for n in onnx.load(str(fused)).graph.node)
    assert ops["MultiHeadAttention"] == (2 if cached else 1) and not ops["Softmax"] and not ops["IsNaN"]
    if cached:  # the cache is still returned, by the same Concat nodes
        assert ops["Concat"] == 2
    cases = [(1, 1, 3), (2, 3, 4), (1, 4, 0)] if cached else [(1, 5, 0), (2, 3, 0)]
    for seed, (batch, seq, past) in enumerate(cases):
        feed = feeds(cached, batch, seq, past, seed)
        for a, b in zip(run(exported, feed), run(fused, feed)):
            np.testing.assert_allclose(a, b, atol=1e-5)  # the causal mask too (several new tokens)


def test_the_fused_graph_is_kept_only_when_it_matches(tmp_path):
    ex = export_module()
    path = tmp_path / "encoder_model.onnx"
    attention_graph(path, cached=False)
    exported = path.read_bytes()
    assert ex.fuse_and_check(path, HEADS, [(1, 5, 0), (2, 3, 0)]).startswith("1 attention block fused")
    assert path.read_bytes() != exported
    assert ex.fuse_and_check(path, HEADS, [(1, 5, 0)]).startswith("no attention block")  # already fused

    attention_graph(path, cached=False)
    real = ex.fuse_attention
    ex.fuse_attention = lambda p, heads: real(p, 1)  # one head of 8 instead of two of 4: other outputs
    message = ex.fuse_and_check(path, HEADS, [(1, 5, 0)])
    assert "outputs differ" in message and message.endswith("kept as exported")
    assert path.read_bytes() == exported and not list(tmp_path.glob("*.unfused"))


def test_where_the_kernel_does_not_support_it_the_export_is_kept(tmp_path):
    """A mask that only broadcasts ([1, 1, 1, 1] here) is fine for the exported Add, not for the kernel's
    attention_bias (it needs every key): the check sees the runtime refuse it and keeps the exported graph."""
    ex = export_module()
    path = tmp_path / "decoder_with_past_model.onnx"
    attention_graph(path, cached=True)
    exported = path.read_bytes()
    message = ex.fuse_and_check(path, HEADS, [(1, 1, 5)])
    assert message.startswith("2 blocks fused but") and message.endswith("kept as exported")
    assert path.read_bytes() == exported


def test_a_graph_without_the_pattern_is_left_as_it_is(tmp_path):
    ex = export_module()
    path = tmp_path / "graph.onnx"
    graph = helper.make_graph([helper.make_node("Softmax", ["x"], ["y"], axis=-1)], "plain",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))
    before = path.read_bytes()
    assert ex.fuse_attention(path, 2) == 0 and path.read_bytes() == before


def test_the_service_says_which_parts_have_it(tmp_path):
    from app.asr import onnx_fused_attention
    from app.models import describe_export

    ex = export_module()
    attention_graph(tmp_path / "encoder_model.onnx", cached=False)
    attention_graph(tmp_path / "decoder_model.onnx", cached=True)
    assert ex.fuse_attention(tmp_path / "encoder_model.onnx", HEADS) == 1  # the decoder stays as exported
    assert onnx_fused_attention(tmp_path / "encoder_model.onnx") and not onnx_fused_attention(tmp_path / "decoder_model.onnx")
    info = describe_export(tmp_path)
    assert info.fused_attention == {"encoder": True, "decoder": False}
    assert info.view(True, None)["fused_attention"] == {"encoder": True, "decoder": False}

