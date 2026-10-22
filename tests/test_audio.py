import shutil
import subprocess

import numpy as np
import pytest

from app import audio
from app.audio import InvalidAudioError, decode_audio, detect_format
from app.features import N_FRAMES, N_SAMPLES, log_mel_spectrogram, mel_filters, pad_or_trim
from tests.conftest import make_wav


def test_detect_format():
    assert detect_format(make_wav()) == "wav"
    assert detect_format(b"ID3\x04\x00") == "mp3"
    assert detect_format(b"\xff\xfb\x90\x00") == "mp3"
    with pytest.raises(InvalidAudioError):
        detect_format(b"%PDF-1.7")


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_decode_wav_resamples_to_16k_mono(monkeypatch, use_ffmpeg):
    if use_ffmpeg and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    if not use_ffmpeg:
        monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    samples = decode_audio(make_wav(seconds=2.0, rate=44100, channels=2))
    assert samples.dtype == np.float32
    assert abs(samples.shape[0] - 32000) <= 2
    assert 0.25 < np.abs(samples).max() <= 0.31


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_mp3(tmp_path):
    mp3 = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", str(mp3)],
        check=True,
    )
    samples = decode_audio(mp3.read_bytes())
    assert abs(samples.shape[0] / 16000 - 1.5) < 0.1


def test_mp3_without_ffmpeg_is_rejected(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    with pytest.raises(InvalidAudioError, match="ffmpeg"):
        decode_audio(b"ID3\x04\x00" + b"\x00" * 100)


def test_log_mel_shape_and_range():
    t = np.arange(N_SAMPLES) / 16000
    mel = log_mel_spectrogram(pad_or_trim((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)))
    assert mel.shape == (80, N_FRAMES)
    assert np.isfinite(mel).all()
    assert mel.max() - mel.min() <= 2.0 + 1e-6  # 8 dB-decades of range, scaled by 1/4


def test_log_mel_peak_follows_frequency():
    t = np.arange(16000) / 16000
    low = log_mel_spectrogram((0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
    high = log_mel_spectrogram((0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32))
    assert low.mean(axis=1).argmax() < high.mean(axis=1).argmax()


def test_mel_filters_shape():
    assert mel_filters(80).shape == (80, 201)
    assert mel_filters(128).shape == (128, 201)
    assert (mel_filters(80) >= 0).all()


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_decode_stops_at_max_seconds(monkeypatch, use_ffmpeg):
    if use_ffmpeg and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    if not use_ffmpeg:
        monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    samples = decode_audio(make_wav(seconds=120, rate=16000), max_seconds=5)
    assert abs(samples.shape[0] / 16000 - 6.0) < 0.05  # limit + 1 s, never the full two minutes


def test_detect_telegram_formats():
    assert detect_format(b"OggS\x00\x02" + b"\x00" * 10) == "ogg"
    assert detect_format(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8) == "mp4"
    assert detect_format(b"fLaC\x00\x00\x00\x22") == "flac"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("codec,ext", [("libopus", "ogg"), ("aac", "m4a"), ("flac", "flac")])
def test_decode_compressed_formats(tmp_path, codec, ext):
    path = tmp_path / f"tone.{ext}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:a", codec, str(path)], check=True)
    samples = decode_audio(path.read_bytes())
    assert abs(samples.shape[0] / 16000 - 1.5) < 0.1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_m4a_with_index_at_the_end_decodes(tmp_path):
    # ffmpeg's default MP4 muxing (like most phones) writes the moov index after the audio data,
    # which a pipe can't seek to; a 30 s clip used to decode to zero samples.
    path = tmp_path / "phone.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                    "-c:a", "aac", "-b:a", "64k", str(path)], check=True)
    data = path.read_bytes()
    assert data.rfind(b"moov") > data.find(b"mdat")
    for source in (path, data):
        assert abs(decode_audio(source).shape[0] / 16000 - 30.0) < 0.1
