"""Audio validation and decoding to 16 kHz mono float32, the input Whisper expects."""

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class InvalidAudioError(ValueError):
    """Raised when an upload is not decodable audio in a supported format."""


# Container formats accepted, with the media type used when serving the original file back.
MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",    # Telegram voice notes (Opus)
    "mp4": "audio/mp4",    # M4A / AAC, Telegram video notes
    "flac": "audio/flac",
}


def detect_format(data: bytes) -> str:
    """Return the container format from the file header, or raise InvalidAudioError."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "mp4"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    raise InvalidAudioError("Unsupported audio format: expected WAV, MP3, OGG/Opus, M4A or FLAC")


def decode_audio(source: bytes | Path, max_seconds: float | None = None, as_int16: bool = False) -> np.ndarray:
    """Decode an audio file (any format in MEDIA_TYPES) into mono float32 at 16 kHz in [-1, 1].

    ffmpeg reads from a file, not a pipe: MP4/M4A written by phones (and Telegram video notes) keep
    their index after the media data, which ffmpeg can only reach on a seekable input.
    With max_seconds, decoding stops shortly after that point, so memory stays bounded no matter
    how long a small, low-bitrate file is; callers detect "too long" from the returned length.
    as_int16 returns the raw 16-bit samples (half the memory); the transcriber converts per window.
    """
    if isinstance(source, Path):
        with source.open("rb") as f:
            fmt = detect_format(f.read(16))
    else:
        fmt = detect_format(source)
    limit = max_seconds + 1 if max_seconds is not None else None
    if shutil.which("ffmpeg"):
        if isinstance(source, Path):
            audio = _decode_ffmpeg(source, limit, as_int16)
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
                tmp.write(source)
                tmp.flush()
                audio = _decode_ffmpeg(Path(tmp.name), limit, as_int16)
    elif fmt == "wav":
        audio = _decode_wav(source.read_bytes() if isinstance(source, Path) else source)
        if limit is not None:
            audio = audio[: int(limit * SAMPLE_RATE)]
        if as_int16:
            audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    else:
        raise InvalidAudioError(f"{fmt.upper()} decoding requires ffmpeg to be installed")
    if audio.size == 0:
        raise InvalidAudioError("Audio file contains no samples")
    return audio


def duration_seconds(audio: np.ndarray) -> float:
    return audio.shape[0] / SAMPLE_RATE


def _decode_ffmpeg(path: Path, limit_seconds: float | None = None, as_int16: bool = False) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if limit_seconds is not None:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise InvalidAudioError(f"Could not decode audio: {detail[-1] if detail else 'ffmpeg failed'}")
    if as_int16:
        return np.frombuffer(proc.stdout, dtype=np.int16)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    audio *= 1 / 32768.0  # in place: avoids a second full-length float copy
    return audio


def _decode_wav(data: bytes) -> np.ndarray:
    """Pure-Python fallback for PCM WAV when ffmpeg is unavailable."""
    try:
        with wave.open(io.BytesIO(data)) as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise InvalidAudioError(f"Could not decode WAV: {e}") from e

    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise InvalidAudioError(f"Unsupported WAV sample width: {width * 8} bit")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and audio.size:
        target_len = int(round(audio.size * SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0, audio.size - 1, target_len), np.arange(audio.size), audio
        ).astype(np.float32)
    return audio
