"""Export a Whisper checkpoint to ONNX in the layout app/asr.py expects.

    pip install -r requirements-export.txt
    python scripts/export_whisper_onnx.py --model openai/whisper-tiny --output models/whisper-tiny

Only needed once, at build time; the API itself needs just ONNX Runtime.

Pipeline (each step keeps the graphs runnable on the runtime's onnxruntime):
  1. optimum export with the KV cache (`automatic-speech-recognition-with-past`): encoder, decoder (first
     call) and decoder_with_past (one token per call).
  2. Untie the LM head: optimum ties proj_out to the token embedding through a Transpose, which ONNX Runtime
     constant-folds into a second fp32 copy at load (and quantize_dynamic leaves fp32 + a per-step
     dequantize). The embedding Gather is rewritten to read the transposed table, so one table serves both.
  3. Fuse the encoder's self-attention into com.microsoft.MultiHeadAttention (no [heads, 1500, 1500]
     score tensor; ~-40 MB peak per window).
  4. INT8 dynamic quantization (weights int8 per channel, activations quantized on the fly).
  5. Merge decoder + decoder_with_past into decoder_model_merged.onnx (shared weights, If node).
  --fp32 skips step 4 (2.5x the RAM, same KV-cache speed).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

KEEP = {"encoder_model.onnx", "decoder_model_merged.onnx"}


def untie_lm_head(path: Path) -> None:
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    producers = {o: n for n in graph.node for o in n.output}
    inits = {t.name: t for t in graph.initializer}
    lm_head = next(n for n in graph.node if n.op_type == "MatMul" and n.output[0] == "logits")
    transpose = producers[lm_head.input[1]]
    if transpose.op_type != "Transpose" or transpose.input[0] not in inits:
        return  # already untied (e.g. exported without weight sharing)
    table = inits[transpose.input[0]]
    weight_t = numpy_helper.to_array(table).T.copy()  # [d_model, vocab]
    graph.initializer.append(numpy_helper.from_array(weight_t, "proj_out.weight_t"))
    lm_head.input[1] = "proj_out.weight_t"
    graph.node.remove(transpose)
    for gather in [n for n in graph.node if n.op_type == "Gather" and n.input[0] == table.name]:
        out = gather.output[0]
        gather.input[0], gather.output[0] = "proj_out.weight_t", out + "_t"
        del gather.attribute[:]
        gather.attribute.append(helper.make_attribute("axis", 1))  # -> [d_model, batch, seq]
        index = list(graph.node).index(gather)
        graph.node.insert(index + 1, helper.make_node("Transpose", [out + "_t"], [out], perm=[1, 2, 0]))
    if not any(table.name in n.input for n in graph.node):
        graph.initializer.remove(table)
    onnx.save(model, path)


def fuse_encoder_attention(path: Path, num_heads: int) -> int:
    """Replace MatMul(q*s, k*s) -> Softmax -> MatMul(v) (optimum's SDPA export) by MultiHeadAttention."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node if n.op_type == "Constant"}
    consts.update({t.name: numpy_helper.to_array(t) for t in graph.initializer})

    def up(name, op):
        node = producers.get(name)
        if node is None or node.op_type != op:
            raise LookupError(f"{name}: expected {op}")
        return node

    def only_consumer(name, op):
        nodes = consumers.get(name, [])
        if len(nodes) != 1 or nodes[0].op_type != op:
            raise LookupError(f"{name}: expected a single {op}")
        return nodes[0]

    fused = 0
    for softmax in [n for n in graph.node if n.op_type == "Softmax"]:
        try:
            qk = up(softmax.input[0], "MatMul")
            mul_q, mul_k = up(qk.input[0], "Mul"), up(qk.input[1], "Mul")
            t_q, t_k = up(mul_q.input[0], "Transpose"), up(mul_k.input[0], "Transpose")
            r_q, r_k = up(t_q.input[0], "Reshape"), up(t_k.input[0], "Reshape")
            av = only_consumer(softmax.output[0], "MatMul")
            t_v = up(av.input[1], "Transpose")
            r_v = up(t_v.input[0], "Reshape")
            t_o = only_consumer(av.output[0], "Transpose")
            r_o = only_consumer(t_o.output[0], "Reshape")
            scale = float(consts[mul_q.input[1]]) * float(consts[mul_k.input[1]])
        except (LookupError, KeyError):
            continue  # not the expected pattern: leave this block unfused
        mha = helper.make_node("MultiHeadAttention", [r_q.input[0], r_k.input[0], r_v.input[0]], [r_o.output[0]],
                               domain="com.microsoft", num_heads=num_heads, scale=scale)
        for n in (r_o, t_o, av, softmax, qk, mul_q, mul_k, t_q, t_k, t_v, r_q, r_k, r_v):
            graph.node.remove(n)
        graph.node.append(mha)
        fused += 1
    if not fused:
        return 0
    # drop the shape-computation nodes that only fed the removed Reshapes, then restore topological order
    needed = {o.name for o in graph.output}
    while True:
        used = needed | {i for n in graph.node for i in n.input}
        dead = [n for n in graph.node if not set(n.output) & used]
        if not dead:
            break
        for n in dead:
            graph.node.remove(n)
    ready = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {""}
    ordered, pending = [], list(graph.node)
    while pending:
        node = next(n for n in pending if all(i in ready for i in n.input))
        ordered.append(node)
        ready.update(node.output)
        pending.remove(node)
    del graph.node[:]
    graph.node.extend(ordered)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(model, path)
    return fused


def quantize(path: Path) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = path.with_suffix(".int8.onnx")
    # QInt8 weights, per output channel. reduce_range (7-bit) is only useful on AVX2/AVX-512 CPUs without VNNI.
    quantize_dynamic(str(path), str(tmp), weight_type=QuantType.QInt8, per_channel=True,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    tmp.replace(path)


def smoke_test(output: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    enc = ort.InferenceSession(str(output / "encoder_model.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(output / "decoder_model_merged.onnx"), providers=["CPUExecutionProvider"])
    n_mels = enc.get_inputs()[0].shape[1]
    hidden = enc.run(None, {enc.get_inputs()[0].name: np.zeros((1, n_mels, 3000), np.float32)})[0]
    names = [o.name for o in dec.get_outputs()]
    past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
            for i in dec.get_inputs() if i.name.startswith("past_key_values.")}
    ids = [50258]
    for step in range(2):
        out = dict(zip(names, dec.run(None, {"input_ids": np.array([ids[-1:] if step else ids], np.int64),
                                             "encoder_hidden_states": hidden,
                                             "use_cache_branch": np.array([step > 0]), **past})))
        for k, v in out.items():
            if k.startswith("present.") and (step == 0 or ".decoder." in k):
                past["past_key_values" + k[len("present"):]] = v
        ids.append(int(out["logits"][0, -1].argmax()))
    assert out["logits"].shape[:2] == (1, 1) and np.isfinite(out["logits"]).all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="openai/whisper-tiny",
                        help="Hugging Face model id, e.g. openai/whisper-base or openai/whisper-tiny.en")
    parser.add_argument("--output", default="models/whisper-tiny")
    parser.add_argument("--fp32", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--prune", action="store_true",
                        help="Delete exported graphs the service does not load (smaller Docker image)")
    args = parser.parse_args()

    if not shutil.which("optimum-cli"):
        print("optimum-cli not found: pip install -r requirements-export.txt", file=sys.stderr)
        return 1

    import json

    from optimum.onnx.graph_transformations import merge_decoders

    output = Path(args.output)
    cmd = ["optimum-cli", "export", "onnx", "--model", args.model,
           "--task", "automatic-speech-recognition-with-past", str(output)]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    for name in ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx", "tokenizer.json"):
        if not (output / name).exists():
            print(f"Export finished but {name} is missing in {output}", file=sys.stderr)
            return 1

    config = json.loads((output / "config.json").read_text())
    for name in ("decoder_model.onnx", "decoder_with_past_model.onnx"):
        untie_lm_head(output / name)
    fused = fuse_encoder_attention(output / "encoder_model.onnx", config["encoder_attention_heads"])
    print(f"fused {fused}/{config['encoder_layers']} encoder attention blocks")
    if not args.fp32:
        for name in ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"):
            quantize(output / name)
    merge_decoders(output / "decoder_model.onnx", output / "decoder_with_past_model.onnx",
                   save_path=output / "decoder_model_merged.onnx", strict=False)
    smoke_test(output)

    if args.prune:
        for path in output.iterdir():
            if path.name.endswith((".onnx", ".onnx_data")) and path.name not in KEEP:
                path.unlink()
                print(f"pruned {path.name}")
    (output / "model_id.txt").write_text(args.model + "\n")  # reported as the model name by the API
    size_mb = sum(f.stat().st_size for f in output.glob("*.onnx*") if f.name in KEEP) / 1e6
    print(f"Exported {args.model} to {output} ({size_mb:.0f} MB of ONNX weights loaded by the API)")
    print(f"Run the API with: WHISPER_MODEL_DIR={output} python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
