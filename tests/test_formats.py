from app.formats import to_srt, to_text, to_vtt
from app.schemas import Transcript, TranscriptSegment


def transcript(*segments):
    return Transcript(
        language="en", duration_seconds=4000.0, text=" ".join(t for _, _, t in segments),
        segments=[TranscriptSegment(index=i, start=s, end=e, text=t) for i, (s, e, t) in enumerate(segments)],
    )


def test_srt():
    t = transcript((0.0, 2.5, "Hello."), (3661.042, 3662.0, "An hour later."))
    assert to_srt(t) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello.\n"
        "\n"
        "2\n01:01:01,042 --> 01:01:02,000\nAn hour later.\n"
    )


def test_vtt():
    t = transcript((0.0, 2.5, "Hello."), (59.999, 61.0, "Next."))
    assert to_vtt(t) == "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello.\n\n00:00:59.999 --> 00:01:01.000\nNext.\n"


def test_text_and_empty_transcript():
    assert to_text(transcript((0.0, 1.0, "Hi"), (1.0, 2.0, "there"))) == "Hi there\n"
    empty = transcript()
    assert to_text(empty) == "" and to_srt(empty) == "" and to_vtt(empty) == "WEBVTT\n"
