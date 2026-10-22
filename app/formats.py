"""Render a transcript as plain text, SubRip (SRT) or WebVTT subtitles."""

from app.schemas import Transcript


def _timestamp(seconds: float, separator: str) -> str:
    millis = round(max(seconds, 0.0) * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_text(transcript: Transcript) -> str:
    return transcript.text + "\n" if transcript.text else ""


def to_srt(transcript: Transcript) -> str:
    blocks = [
        f"{n}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{s.text}\n"
        for n, s in enumerate(transcript.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    blocks = [f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{s.text}\n" for s in transcript.segments]
    return "\n".join(["WEBVTT\n", *blocks])


RENDERERS = {
    "text": (to_text, "text/plain; charset=utf-8"),
    "srt": (to_srt, "application/x-subrip; charset=utf-8"),
    "vtt": (to_vtt, "text/vtt; charset=utf-8"),
}
