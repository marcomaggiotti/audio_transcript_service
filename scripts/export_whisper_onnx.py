"""Export a Whisper checkpoint to ONNX in the layout app/asr.py expects.

    pip install -r requirements-export.txt
    python scripts/export_whisper_onnx.py --model openai/whisper-tiny.en --output models/whisper-tiny.en

Only needed once, at build time; the API itself needs just ONNX Runtime.

Pipeline (each step keeps the graphs runnable on the runtime's onnxruntime):
  1. optimum export with the KV cache (`automatic-speech-recognition-with-past`): encoder, decoder (first
     call) and decoder_with_past (one token per call).
  2. Untie the LM head: optimum ties proj_out to the token embedding through a Transpose, which ONNX Runtime
     constant-folds into a second fp32 copy at load (and quantize_dynamic leaves fp32 + a per-step
     dequantize). The embedding Gather is rewritten to read the transposed table, so one table serves both.
  3. Fuse every attention block into com.microsoft.MultiHeadAttention, ONNX Runtime's transformer kernel: the
     encoder's self-attention (no [heads, 1500, 1500] score tensor; ~-40 MB peak per window) and the decoders'
     self-attention (the causal mask passed as attention_bias, the KV cache as it is) and cross-attention. Where
     the graph doesn't have the expected pattern a block is left as it is, and a fused graph is kept only if it
     gives the same outputs as the exported one (checked on sample inputs; otherwise the export's is kept).
     whisper-tiny: 20 blocks; a cached decoder step 9% faster (int8) and 4% (fp32), the same transcripts.
     --no-fuse-attention skips it. (ONNX Runtime's own transformer optimizer finds no attention in this export,
     and its LayerNorm/GELU fusions are what the runtime's graph optimization already does when it opens a model.)
  4. INT8 dynamic quantization of the decoder (weights int8 per channel, activations quantized on the fly).
     The encoder stays fp32: quantizing it garbles real speech. Measured with the real whisper-tiny weights
     (20 TTS sentences + a recorded voice note): fp32 5.0% WER; decoder int8 5.7% (+27 MB RAM over all-int8);
     encoder int8 7.1%, and the voice note "Take a bottle of milk and kilogram of flour." became
     "Thank you. A little bit of milk and I'll take you to the flower.".
  5. Merge decoder + decoder_with_past into decoder_model_merged.onnx (shared weights, If node).
  With --variants (the default) the other precisions of each part are kept too, as encoder_model.<p>.onnx and
  decoder_model_merged.<p>.onnx, so a job can ask for them (POST /jobs encoder_precision / decoder_precision):
  fp32, int8, and int4 (4-bit weights in blocks of 32, MatMulNBits: about a quarter of the fp32 size; --no-int4).
  6. With --ct2, the same checkpoint converted for CTranslate2 into OUTPUT/ct2 (POST /jobs engine=ctranslate2).
  --quantize all also quantizes the encoder (smallest, worse transcripts); --quantize none (or --fp32) skips
  step 4 (about +90 MB of RAM for tiny, a little more accurate).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

KEEP = {"encoder_model.onnx", "decoder_model_merged.onnx",  # the defaults the service loads
        # the other precisions of each part, loaded only by jobs that ask for them (--variants)
        "encoder_model.fp32.onnx", "encoder_model.int8.onnx", "encoder_model.int4.onnx",
        "decoder_model_merged.fp32.onnx", "decoder_model_merged.int8.onnx", "decoder_model_merged.int4.onnx"}
INT4_BLOCK = 32  # weights quantized in blocks of 32 along the input dimension, each with its own scale


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


def fuse_attention(path: Path, num_heads: int) -> int:
    """Replace each attention block of optimum's (SDPA) export by com.microsoft.MultiHeadAttention:

        q: Reshape(q) -> Transpose(0,2,1,3) -> Mul(c_q)                 (q already scaled by the export)
        k: Reshape(k) -> Transpose(0,2,3,1) -> Mul(c_k)   or   K [batch, heads, L, head] -> Transpose(0,1,3,2) -> Mul(c_k)
        MatMul(q, k) [-> Add(mask)] -> Softmax [-> Where(IsNaN(p), 0, p)] -> MatMul(., V) -> Transpose(0,2,1,3) -> Reshape

    becomes MultiHeadAttention(q, key, value, attention_bias=mask, scale=c_q*c_k), with key and value as
    [batch, L, hidden] (projections: the encoder, a decoder's first call) or [batch, heads, L, head] (the KV cache:
    later calls). The causal mask goes in unchanged, so any number of new tokens is masked as before. Returns how
    many blocks were fused; one that doesn't match is left as exported."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node
              if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value"}
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

    def perm(node):
        return next((list(a.ints) for a in node.attribute if a.name == "perm"), None)

    def scalar(name):
        value = consts[name]
        if value.size != 1:
            raise LookupError(f"{name}: not a scalar")
        return float(value.reshape(-1)[0])

    fused = 0
    for softmax in [n for n in graph.node if n.op_type == "Softmax"]:
        try:
            core, scores, mask = [], softmax.input[0], None  # core: the nodes this block replaces
            add = producers.get(scores)
            if add is not None and add.op_type == "Add":  # scores + causal mask
                scores = next(i for i in add.input if producers.get(i) is not None and producers[i].op_type == "MatMul")
                mask = next(i for i in add.input if i != scores)
                core.append(add)
            qk = up(scores, "MatMul")
            mul_q, mul_k = up(qk.input[0], "Mul"), up(qk.input[1], "Mul")
            t_q, t_k = up(mul_q.input[0], "Transpose"), up(mul_k.input[0], "Transpose")
            r_q = up(t_q.input[0], "Reshape")
            if perm(t_q) != [0, 2, 1, 3]:
                raise LookupError("query layout")
            probs, users = softmax.output[0], consumers.get(softmax.output[0], [])
            if len(users) == 2 and sorted(u.op_type for u in users) == ["IsNaN", "Where"]:  # rows with no key: 0
                isnan = next(u for u in users if u.op_type == "IsNaN")
                where = only_consumer(isnan.output[0], "Where")
                if where.input[2] != probs:
                    raise LookupError("NaN guard")
                core += [isnan, where]
                probs = where.output[0]
            av = only_consumer(probs, "MatMul")
            t_o = only_consumer(av.output[0], "Transpose")
            r_o = only_consumer(t_o.output[0], "Reshape")
            if perm(t_o) != [0, 2, 1, 3]:
                raise LookupError("output layout")
            if perm(t_k) == [0, 2, 3, 1]:  # keys from a projection: key and value as [batch, L, hidden]
                key = up(t_k.input[0], "Reshape").input[0]
                t_v = up(av.input[1], "Transpose")
                if perm(t_v) != [0, 2, 1, 3]:
                    raise LookupError("value layout")
                value = up(t_v.input[0], "Reshape").input[0]
            elif perm(t_k) == [0, 1, 3, 2]:  # keys from the cache: key and value as [batch, heads, L, head]
                key, value = t_k.input[0], av.input[1]
            else:
                raise LookupError("key layout")
            scale = scalar(mul_q.input[1]) * scalar(mul_k.input[1])
        except (LookupError, KeyError, StopIteration):
            continue  # not the expected pattern: leave this block as exported
        name = f"{softmax.name.rsplit('/', 1)[0]}/MultiHeadAttention" if softmax.name else f"MultiHeadAttention_{fused}"
        mha = helper.make_node("MultiHeadAttention", [r_q.input[0], key, value] + (["", "", mask] if mask else []),
                               [r_o.output[0]], name=name, domain="com.microsoft", num_heads=num_heads, scale=scale)
        for n in core + [qk, mul_q, mul_k, softmax, av, t_o, r_o]:
            graph.node.remove(n)
        graph.node.append(mha)
        fused += 1
    if not fused:
        return 0
    # drop what only fed the replaced nodes (reshapes, transposes, shape computations), then restore topological order
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


def sample_inputs(path: Path, batch: int = 1, sequence: int = 3, past: int = 5, seed: int = 0) -> dict:
    """Random inputs for one of the exported graphs, the dimensions by their names: token ids, encoder states, a
    KV cache of `past` tokens, `sequence` new tokens (the encoder's mel features have fixed dimensions)."""
    import numpy as np
    import onnx

    rng = np.random.default_rng(seed)
    feed = {}
    for inp in onnx.load(path, load_external_data=False).graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            name = d.dim_param
            dims.append(d.dim_value if d.dim_value else batch if "batch" in name else past if "past" in name
                        else 1500 if "encoder_sequence" in name else sequence)
        if inp.name == "input_ids":
            feed[inp.name] = rng.integers(0, 50000, size=dims, dtype=np.int64)
        else:
            feed[inp.name] = rng.standard_normal(dims).astype(np.float32)
    return feed


def max_difference(original: Path, changed: Path, feeds: list[dict]) -> float:
    """The largest difference between two graphs' outputs on these inputs (ONNX Runtime, CPU)."""
    import numpy as np
    import onnxruntime as ort

    a = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(changed), providers=["CPUExecutionProvider"])
    worst = 0.0
    for feed in feeds:
        for x, y in zip(a.run(None, feed), b.run(None, feed)):
            if x.size:
                diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                if not np.isfinite(diff):
                    return float("nan")
                worst = max(worst, diff)
    return worst


def fuse_and_check(path: Path, num_heads: int, feeds: list[tuple[int, int, int]], tolerance: float = 1e-3) -> str:
    """fuse_attention, kept only if the fused graph runs and gives the exported graph's outputs (within
    `tolerance`) on sample inputs: (batch, new tokens, cached tokens) per case. Says what happened."""
    original = path.with_name(path.name + ".unfused")
    shutil.copy(path, original)
    try:
        fused = fuse_attention(path, num_heads)
        if not fused:
            return "no attention block in the expected pattern: kept as exported"
        cases = [sample_inputs(original, b, s, p, seed) for seed, (b, s, p) in enumerate(feeds)]
        try:
            diff = max_difference(original, path, cases)
        except Exception as e:  # the runtime can't run it (an older ONNX Runtime, another layout)
            diff, why = float("nan"), f"{type(e).__name__}: {str(e)[:200]}"
        else:
            why = f"outputs differ by {diff:g}"
        if not diff <= tolerance:
            shutil.copy(original, path)
            return f"{fused} block{'s' if fused != 1 else ''} fused but {why}: kept as exported"
        return (f"{fused} attention block{'s' if fused != 1 else ''} fused ("
                + (f"outputs within {diff:.1g} of" if diff else "the same outputs as") + " the exported graph)")
    finally:
        original.unlink(missing_ok=True)


def quantize(path: Path) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = path.with_name(path.name + ".quantizing")
    # QInt8 weights, per output channel. reduce_range (7-bit) is only useful on AVX2/AVX-512 CPUs without VNNI.
    quantize_dynamic(str(path), str(tmp), weight_type=QuantType.QInt8, per_channel=True,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    tmp.replace(path)


def quantize_int4(path: Path) -> None:
    """4-bit weights for every MatMul with a constant weight (MatMulNBits, blocks of INT4_BLOCK, symmetric):
    about a quarter of the fp32 size; activations stay fp32. Embeddings and convolutions are left as they are."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(path))
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=INT4_BLOCK, is_symmetric=True)
    quantizer.process()
    tmp = path.with_name(path.name + ".quantizing")
    quantizer.model.save_model_to_file(str(tmp), use_external_data_format=False)
    tmp.replace(path)


def convert_ct2(model: str, output: Path) -> None:
    """The same checkpoint for CTranslate2 (faster-whisper): float16 weights (converted again to the job's compute
    type on load), with the tokenizer and preprocessor config faster-whisper reads from the folder."""
    cmd = ["ct2-transformers-converter", "--model", model, "--output_dir", str(output), "--quantization", "float16",
           "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    (output / "model_id.txt").write_text(model + "\n")


def build_precisions(output: Path, default: dict[str, str], variants: bool, int4: bool = True) -> None:
    """encoder_model.onnx / decoder_model_merged.onnx in the default precision, and with `variants` the other
    precisions as encoder_model.<p>.onnx / decoder_model_merged.<p>.onnx. INT8 and INT4 graphs are quantized copies
    of the fp32 ones (INT4 only with `int4`)."""
    from optimum.onnx.graph_transformations import merge_decoders

    wanted = {part: ({"fp32", "int8"} | ({"int4"} if int4 else set())) if variants else {default[part]} for part in default}
    encoder = output / "encoder_model.onnx"
    if "int4" in wanted["encoder"]:
        shutil.copy(encoder, output / "encoder_model.int4.onnx")  # from the fp32 graph, before any default swap
        quantize_int4(output / "encoder_model.int4.onnx")
    if "int8" in wanted["encoder"]:
        shutil.copy(encoder, output / "encoder_model.int8.onnx")
        quantize(output / "encoder_model.int8.onnx")
    if default["encoder"] == "int8":
        if variants:
            shutil.copy(encoder, output / "encoder_model.fp32.onnx")
        (output / "encoder_model.int8.onnx").replace(encoder)

    decoders = ("decoder_model", "decoder_with_past_model")
    for precision in sorted(wanted["decoder"]):
        names = [f"{d}.onnx" for d in decoders]
        if precision in ("int8", "int4"):
            for d in decoders:
                shutil.copy(output / f"{d}.onnx", output / f"{d}.{precision}.onnx")
                (quantize if precision == "int8" else quantize_int4)(output / f"{d}.{precision}.onnx")
            names = [f"{d}.{precision}.onnx" for d in decoders]
        target = "decoder_model_merged.onnx" if precision == default["decoder"] else f"decoder_model_merged.{precision}.onnx"
        merge_decoders(output / names[0], output / names[1], save_path=output / target, strict=False)
        if precision in ("int8", "int4"):
            for name in names:
                (output / name).unlink()


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
    config = json.loads((output / "config.json").read_text()) if (output / "config.json").exists() else {}
    ids = [config.get("decoder_start_token_id") or 50257]  # <|startoftranscript|> of this vocabulary
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
    parser.add_argument("--model", default="openai/whisper-tiny.en",
                        help="Hugging Face id of an English-only model, e.g. openai/whisper-tiny.en or openai/whisper-base.en")
    parser.add_argument("--output", default="models/whisper-tiny.en")
    parser.add_argument("--quantize", choices=["decoder", "all", "none"], default="decoder",
                        help="INT8 weights for: the decoder only (default), encoder and decoder (smaller, but "
                             "garbles real speech), or nothing")
    parser.add_argument("--fp32", action="store_true", help="Same as --quantize none")
    parser.add_argument("--no-variants", dest="variants", action="store_false",
                        help="Only the default precision of each part (smaller image; jobs can't choose)")
    parser.add_argument("--no-int4", dest="int4", action="store_false",
                        help="With variants, no 4-bit copies (encoder_model.int4.onnx, decoder_model_merged.int4.onnx)")
    parser.add_argument("--ct2", action="store_true",
                        help="Also convert the checkpoint for CTranslate2 into OUTPUT/ct2 (needs ctranslate2), for "
                             "jobs with engine=ctranslate2")
    parser.add_argument("--no-fuse-attention", dest="fuse_attention", action="store_false",
                        help="Keep the attention as exported (no MultiHeadAttention), e.g. to measure the fusion")
    parser.add_argument("--prune", action="store_true",
                        help="Delete exported graphs the service does not load (smaller Docker image)")
    args = parser.parse_args()

    if not shutil.which("optimum-cli"):
        print("optimum-cli not found: pip install -r requirements-export.txt", file=sys.stderr)
        return 1

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
    if args.fuse_attention:  # (batch, new tokens, cached tokens) of the sample inputs each graph is checked on
        for name, heads, feeds in (("encoder_model.onnx", config["encoder_attention_heads"], [(1, 1, 0)]),
                                   ("decoder_model.onnx", config["decoder_attention_heads"], [(1, 4, 0), (2, 3, 0)]),
                                   ("decoder_with_past_model.onnx", config["decoder_attention_heads"],
                                    [(1, 1, 5), (2, 3, 7)])):
            print(f"{name}: {fuse_and_check(output / name, heads, feeds)}")
    mode = "none" if args.fp32 else args.quantize
    default = {"encoder": "int8" if mode == "all" else "fp32", "decoder": "fp32" if mode == "none" else "int8"}
    build_precisions(output, default, args.variants, args.int4)
    print(f"default precision: encoder {default['encoder']}, decoder {default['decoder']}"
          + ((" (fp32, int8" + (" and int4" if args.int4 else "") + " kept)") if args.variants else ""))
    smoke_test(output)

    if args.prune:
        for path in output.iterdir():
            if path.name.endswith((".onnx", ".onnx_data")) and path.name not in KEEP:
                path.unlink()
                print(f"pruned {path.name}")
    (output / "model_id.txt").write_text(args.model + "\n")  # reported as the model name by the API
    if args.ct2:
        convert_ct2(args.model, output / "ct2")
    size = lambda names: sum(f.stat().st_size for f in output.glob("*.onnx*") if f.name in names) / 1e6  # noqa: E731
    loaded = size({"encoder_model.onnx", "decoder_model_merged.onnx"})
    print(f"Exported {args.model} to {output} ({loaded:.0f} MB of ONNX weights loaded by the API"
          + (f", plus {size(KEEP) - loaded:.0f} MB of other precisions loaded on request)" if args.variants else ")"))
    print(f"Run the API with: WHISPER_MODEL_DIR={output} python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
