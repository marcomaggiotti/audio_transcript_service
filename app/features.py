"""Whisper log-mel spectrogram in NumPy.

Matches openai/whisper `log_mel_spectrogram` (and the HF WhisperFeatureExtractor):
25 ms Hann window, 10 ms hop, Slaney mel filterbank, log10 + dynamic range clamp.
Implemented here so inference needs only NumPy + ONNX Runtime (no torch/librosa).
"""

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
N_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE  # 480000 samples per window
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames per window
SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(freq >= min_log_hz, min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


@lru_cache
def mel_filters(n_mels: int = 80) -> np.ndarray:
    """Slaney-normalised mel filterbank, shape (n_mels, N_FFT // 2 + 1)."""
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(SAMPLE_RATE / 2)), n_mels + 2)
    mel_f = _mel_to_hz(mel_points)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    return (weights * enorm[:, None]).astype(np.float32)


@lru_cache
def _hann_window() -> np.ndarray:
    # Periodic Hann window, as torch.hann_window(N_FFT) produces.
    n = np.arange(N_FFT)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if audio.shape[0] > length:
        return audio[:length]
    if audio.shape[0] < length:
        return np.pad(audio, (0, length - audio.shape[0]))
    return audio


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Return a (n_mels, frames) float32 log-mel spectrogram for 16 kHz mono audio."""
    audio = audio.astype(np.float32)
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (padded.shape[0] - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
        writeable=False,
    )
    stft = np.fft.rfft(frames * _hann_window(), axis=1)
    magnitudes = (np.abs(stft[:-1]) ** 2).T  # drop the last frame like whisper does

    mel_spec = mel_filters(n_mels) @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
