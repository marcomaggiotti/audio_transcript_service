"""The encoder cache and the shared encoder sessions (app/encoder_cache.py): outputs kept in memory and on disk, reused
by the same encoder for the same window, by any model with the same encoder file; sessions shared the same way."""

import gc
import json
import os
import time

import numpy as np
from fastapi.testclient import TestClient

from app.asr import TranscriptionResult, WindowTrace
from app.encoder_cache import EncoderCache, SessionPool, file_digest, mel_digest
from app.main import Components, create_app
from app.models import ModelSet
from tests.conftest import FakeTranscriber, make_wav, wait_for
from tests.test_asr import WORDS, make, ts
from tests.test_models import export


def mb(n: float, value: float = 0.0) -> np.ndarray:
    return np.full(int(n * 1e6 / 4), value, dtype=np.float32)


def test_memory_keeps_the_most_recently_used_within_the_limit():
    cache = EncoderCache(limit_mb=2.5)
    for key in "abc":
        cache.put(key, mb(1, ord(key)))
    assert cache.get("a") is None and cache.get("b") is not None  # a went first; b is now the most recent
    cache.put("d", mb(1))
    assert cache.get("c") is None and cache.get("b") is not None and cache.get("d") is not None
    kept = cache.get("b")
    assert not kept.flags.writeable and cache.get("b") is kept  # read-only, the same array
    assert cache.put("huge", mb(3)) is not None and cache.get("huge") is None  # bigger than the limit: not kept
    stats = cache.stats()
    assert stats["memory"]["windows"] == 2 and stats["memory"]["used_mb"] == 2.0 and stats["misses"] == 3
    assert not EncoderCache(limit_mb=0).enabled


def test_disk_keeps_them_across_restarts_and_within_its_limit(tmp_path):
    first = EncoderCache(limit_mb=0, disk_dir=tmp_path, disk_limit_mb=2.5)
    first.put("aa", mb(1, 1.0))
    restarted = EncoderCache(limit_mb=10, disk_dir=tmp_path, disk_limit_mb=2.5)
    found = restarted.get("aa")
    assert found is not None and float(found[0]) == 1.0 and restarted.stats()["disk_hits"] == 1
    assert restarted.get("aa") is found  # now in memory too
    for i, key in enumerate(["bb", "cc"]):
        time.sleep(0.02)  # distinct modification times: the oldest is deleted first
        restarted.put(key, mb(1, float(i)))
    files = sorted(p.stem for p in tmp_path.glob("*.npy"))
    assert files == ["bb", "cc"] and restarted.stats()["disk"]["used_mb"] <= 2.5
    (tmp_path / "zz.tmp").write_text("cut off")
    EncoderCache(limit_mb=1, disk_dir=tmp_path, disk_limit_mb=5)
    assert not (tmp_path / "zz.tmp").exists()  # a write cut off by a restart is cleaned up


def test_digests_follow_the_content(tmp_path):
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_bytes(b"encoder weights")
    b.write_bytes(b"encoder weights")
    assert file_digest(a) == file_digest(b)  # the same file in two folders: one encoder
    time.sleep(0.01)
    b.write_bytes(b"encoder weights, trained")
    os.utime(b, ns=(time.time_ns(), time.time_ns()))
    assert file_digest(a) != file_digest(b)
    mel = np.random.default_rng(0).standard_normal((80, 3000)).astype(np.float32)
    assert mel_digest(mel) == mel_digest(mel.copy()) != mel_digest(mel + 1e-6)


def test_the_pool_shares_a_session_while_it_is_in_use():
    class Session:
        pass
    pool, made = SessionPool(), []
    create = lambda: made.append(Session()) or made[-1]  # noqa: E731
    first, shared = pool.open(("enc", "abc"), create)
    again, shared_again = pool.open(("enc", "abc"), create)
    assert again is first and (shared, shared_again) == (False, True) and len(made) == 1
    other, _ = pool.open(("enc", "def"), create)
    assert other is not first and pool.live() == 2
    del first, again, made[:]
    gc.collect()
    assert pool.live() == 1  # freed when no model uses it: the pool holds it weakly
    assert pool.open(("enc", "abc"), create)[1] is False


def test_the_same_window_isn_t_encoded_twice(tmp_path):
    encoder_file = tmp_path / "encoder_model.onnx"
    encoder_file.write_bytes(b"the encoder")
    cache = EncoderCache(limit_mb=64)
    script = [ts(0), 0, 1, ts(2.0), ts(2.0), WORDS.index("bye"), ts(3.0)]
    audio = np.random.default_rng(1).standard_normal(16000 * 5).astype(np.float32) * 0.1

    def model():
        t = make([script])
        t.encoder_cache, t._encoder_source = cache, (encoder_file, "all")
        return t
    first, second = model(), model()
    a = first.transcribe(audio)
    b = second.transcribe(audio)  # another model with the same encoder file: the output is reused
    assert first.encoder.calls == 1 and second.encoder.calls == 0
    assert [w.encoder_cached for w in a.windows] == [False] and [w.encoder_cached for w in b.windows] == [True]
    assert a.text == b.text == "hello world bye"

    third = model()
    third._encoder_source = (encoder_file, "basic")  # another graph optimization level: its own outputs
    third.transcribe(audio)
    assert third.encoder.calls == 1
    plain = make([script])  # no cache (test stand-ins, CTranslate2): encoded as before
    plain.transcribe(audio)
    assert plain.encoder.calls == 1 and cache.stats()["hits"] == 1


class WindowedTranscriber(FakeTranscriber):
    def transcribe(self, audio, on_progress=None, options=None):
        result = super().transcribe(audio, on_progress, options)
        return TranscriptionResult(segments=result.segments, windows=[
            WindowTrace(0.0, 30.0, 28.0, 0.01, False, 2, encoder_cached=True),
            WindowTrace(28.0, 2.0, 2.0, 0.02, False, 1)])


def test_jobs_say_how_many_windows_came_from_the_cache(settings):
    with TestClient(create_app(settings, Components(transcriber=WindowedTranscriber()))) as client:
        job = client.post("/jobs", files={"file": ("a.wav", make_wav(1.0), "audio/wav")}).json()
        metrics = wait_for(client, job["id"])["metrics"]
        assert (metrics["windows"], metrics["encoder_cache_hits"]) == (2, 1)
        windows = client.get(f"/jobs/{job['id']}/transcript").json()["windows"]
        assert [w["encoder_cached"] for w in windows] == [True, False]
        cache = client.get("/config").json()["encoder_cache"]
        assert cache["enabled"] and cache["memory"]["limit_mb"] == 64.0 and cache["disk"] is None


def test_a_model_with_the_service_s_encoder_shares_it(tmp_path):
    models_dir = tmp_path / "models"
    default = export(models_dir, "whisper-tiny.en", "openai/whisper-tiny.en", size=40_000_000)
    tuned = export(models_dir, "tiny-words", "finetuned/tiny-words", size=40_000_000)  # its encoder file, byte for byte
    (tuned / "decoder_model_merged.onnx").write_bytes(b"MatMulInteger, trained")
    other = export(models_dir, "whisper-base.en", "openai/whisper-base.en", size=80_000_000)
    service_model = FakeTranscriber()
    service_model.name = "openai/whisper-tiny.en"
    models = ModelSet(service_model, str(default), str(models_dir), lambda p: None, limit_mb=512)
    by_id = {m.id: m for m in models.list()}
    assert by_id["finetuned/tiny-words"].shared_encoder == "openai/whisper-tiny.en"
    assert by_id["openai/whisper-base.en"].shared_encoder is None
    # The memory check counts only the decoder of a model that shares the (loaded) encoder.
    assert by_id["finetuned/tiny-words"].needs_mb < by_id["openai/whisper-base.en"].needs_mb
    assert by_id["finetuned/tiny-words"].needs_mb == round(len(b"MatMulInteger, trained") / 1e6 * 1.3 + 150)
    added = export(tmp_path / "later", "tiny-later", "finetuned/tiny-later", size=40_000_000)
    assert models.add(added).shared_encoder == "openai/whisper-tiny.en"
    view = models.add(added).view(True, None)
    assert view["shared_encoder"] == "openai/whisper-tiny.en"
    assert json.loads(json.dumps(view))  # serializable for GET /config
    del other
