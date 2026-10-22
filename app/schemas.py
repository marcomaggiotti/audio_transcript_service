"""API models."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class OutputFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"


class TranscriptSegment(BaseModel):
    index: int
    start: float = Field(description="Seconds from the start of the audio")
    end: float
    text: str
    avg_logprob: float | None = Field(None, description="Mean log-probability of the segment's tokens")
    no_speech_prob: float | None = Field(None, description="Whisper's estimate that the 30 s window is silence")


class InferenceMetrics(BaseModel):
    audio_seconds: float
    decode_seconds: float
    inference_seconds: float
    real_time_factor: float = Field(description="Inference time / audio length; below 1 is faster than real time")
    peak_memory_mb: float


class Transcript(BaseModel):
    language: str | None
    duration_seconds: float
    text: str
    segments: list[TranscriptSegment]


class Job(BaseModel):
    id: str
    filename: str
    size_bytes: int
    language: str | None = Field(None, description="Requested language; None = auto-detect")
    status: JobStatus = JobStatus.queued
    stage: str | None = Field(None, description="While processing: decoding, then transcribing")
    progress: float = Field(0.0, ge=0, le=1, description="Share of the audio transcribed so far (0-1)")
    queue_position: int | None = Field(None, description="Jobs ahead of this one while it is queued (0 = next)")
    error: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: InferenceMetrics | None = None
    transcript: Transcript | None = None


class JobSummary(BaseModel):
    id: str
    filename: str
    status: JobStatus
    status_url: str
    transcript_url: str


class TranscriptResponse(Transcript):
    model: str | None
    metrics: InferenceMetrics | None


class HealthResponse(BaseModel):
    status: str
    model: str | None
    model_error: str | None = None
    languages: list[str]
    queue_depth: int
    active_jobs: int
