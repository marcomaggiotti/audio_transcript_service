"""A tiny random English-only Whisper (d_model 32, one layer each side), saved like a Hugging Face checkpoint without
downloading anything: its vocabulary is the 256 byte tokens, a few merges, Whisper's special tokens and the 1501
timestamp tokens. For the training tests (needs torch and transformers); its transcripts are noise."""

import json
from pathlib import Path

from transformers import (GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration,
                          WhisperTokenizer, WhisperTokenizerFast)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|translate|>", "<|transcribe|>", "<|startoflm|>",
            "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"]


def make_tiny(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    merges = []
    for a, b in [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("e", "r"), ("o", "n")]:
        vocab.setdefault(a + b, len(vocab))
        merges.append(f"{a} {b}")
    for t in SPECIALS + [f"<|{i * 0.02:.2f}|>" for i in range(1501)]:
        vocab[t] = len(vocab)
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n")
    slow = WhisperTokenizer(str(out / "vocab.json"), str(out / "merges.txt"), unk_token="<|endoftext|>",
                            bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                            additional_special_tokens=SPECIALS[1:])
    slow.save_pretrained(out)
    WhisperTokenizerFast.from_pretrained(out).save_pretrained(out)  # tokenizer.json, as the real checkpoints have
    WhisperFeatureExtractor(feature_size=80).save_pretrained(out)
    ids = {t: vocab[t] for t in SPECIALS}
    eot, sot = ids["<|endoftext|>"], ids["<|startoftranscript|>"]
    config = WhisperConfig(vocab_size=len(vocab), num_mel_bins=80, d_model=32, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                           max_source_positions=1500, max_target_positions=448, pad_token_id=eot, bos_token_id=eot,
                           eos_token_id=eot, decoder_start_token_id=sot)
    model = WhisperForConditionalGeneration(config)
    model.generation_config = GenerationConfig(
        decoder_start_token_id=sot, bos_token_id=eot, eos_token_id=eot, pad_token_id=eot,
        no_timestamps_token_id=ids["<|notimestamps|>"], prev_sot_token_id=ids["<|startofprev|>"], is_multilingual=False,
        max_initial_timestamp_index=50, max_length=448, suppress_tokens=[], begin_suppress_tokens=[eot],
        return_timestamps=False)
    model.save_pretrained(out)
    return out
