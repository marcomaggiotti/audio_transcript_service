"""Audio Transcript Service API: upload audio, get its text back from Whisper running on ONNX Runtime."""

import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.asr import OnnxWhisperTranscriber, Transcriber
from app.audio import InvalidAudioError, detect_format
from app.config import Settings, get_settings
from app.formats import RENDERERS
from app.jobs import JobNotFound, QueueFullError, TranscriptionService
from app.schemas import (
    HealthResponse,
    Job,
    JobStatus,
    JobSummary,
    OutputFormat,
    TranscriptResponse,
)

logger = logging.getLogger(__name__)

UPLOAD_CHUNK = 1024 * 1024
MULTIPART_SLACK = 1024 * 1024  # form fields + boundaries on top of the file itself
UPLOAD_PATHS = ("/transcribe", "/jobs")
# Guarded when API_TOKEN is set. Health checks and the API docs stay reachable.
PROTECTED_PREFIXES = ("/transcribe", "/jobs", "/languages")
LANGUAGE_PATTERN = r"^([a-z]{2,3}|auto)$"


def _is_protected(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


def _token_ok(request: Request, expected: str) -> bool:
    """Accept `X-API-Key: <token>` or `Authorization: Bearer <token>` (scheme case-insensitive)."""
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    bearer = credentials.strip() if scheme.lower() == "bearer" else ""
    supplied = request.headers.get("x-api-key", "").strip() or bearer
    return hmac.compare_digest(supplied.encode(), expected.encode())


@dataclass
class Components:
    """Injectable dependencies; tests pass fakes, production builds them from Settings."""

    transcriber: Transcriber | None = None


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    settings = settings or get_settings()
    components = components or Components()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transcriber, transcriber_error = components.transcriber, None
        if transcriber is None:
            try:
                transcriber = OnnxWhisperTranscriber.from_dir(
                    settings.whisper_model_dir, settings.max_new_tokens, settings.onnx_threads,
                    low_memory=settings.whisper_low_memory,
                )
                logger.info("Loaded speech model %s", transcriber.name)
            except Exception as e:  # keep serving: /health reports the reason, /healthz fails
                transcriber_error = str(e)
                logger.error("Speech model not loaded: %s", e)

        service = TranscriptionService(
            transcriber, settings.upload_dir, transcriber_error,
            max_audio_seconds=settings.max_audio_seconds,
            concurrency=settings.worker_concurrency,
            max_queue_size=settings.max_queue_size,
            max_finished_jobs=settings.max_finished_jobs,
        )
        app.state.service = service
        await service.start()
        yield
        await service.stop()

    app = FastAPI(
        title="Audio Transcript Service",
        description="Transcribe audio to text with Whisper on ONNX Runtime: plain text, timestamped "
                    "segments, SRT or WebVTT.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Reject unauthenticated or oversized requests before the request body is read or spooled."""
        method, path = request.method, request.url.path.rstrip("/")
        if settings.api_token and _is_protected(path) and not _token_ok(request, settings.api_token):
            return JSONResponse({"detail": "Missing or invalid API token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        if method == "POST" and path in UPLOAD_PATHS:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + MULTIPART_SLACK:
                return JSONResponse({"detail": f"File exceeds {settings.max_upload_mb} MB limit"}, status_code=413)
        return await call_next(request)

    def service_of(request: Request) -> TranscriptionService:
        return request.app.state.service

    def get_job(service: TranscriptionService, job_id: str) -> Job:
        try:
            return service.get(job_id)
        except JobNotFound:
            raise HTTPException(404, f"Job {job_id} not found")

    async def accept_upload(service: TranscriptionService, file: UploadFile, language: str | None) -> Job:
        """Validate and store the upload, then queue it as a job."""
        if service.transcriber is None:
            raise HTTPException(503, f"Speech model unavailable: {service.transcriber_error or 'not loaded'}")
        language = language or settings.default_language
        if language != "auto" and service.languages and language not in service.languages:
            raise HTTPException(422, f"Unsupported language '{language}'; see GET /languages")

        job_id = uuid.uuid4().hex
        path = service.upload_path(job_id)
        size = 0
        try:
            with path.open("wb") as out:
                while chunk := await file.read(UPLOAD_CHUNK):
                    if size == 0:
                        detect_format(chunk)
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "Empty file")
        except InvalidAudioError as e:
            path.unlink(missing_ok=True)
            raise HTTPException(415, str(e))
        except HTTPException:
            path.unlink(missing_ok=True)
            raise

        job = Job(id=job_id, filename=file.filename or "audio", size_bytes=size,
                  language=None if language == "auto" else language)
        try:
            service.submit(job)
        except QueueFullError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"})
        return job

    def render(job: Job, fmt: OutputFormat):
        if fmt == OutputFormat.json:
            return TranscriptResponse(**job.transcript.model_dump(), model=job.model, metrics=job.metrics)
        renderer, media_type = RENDERERS[fmt.value]
        stem = Path(job.filename).stem or "transcript"
        extension = "txt" if fmt == OutputFormat.text else fmt.value
        return PlainTextResponse(renderer(job.transcript), media_type=media_type,
                                 headers={"Content-Disposition": f'inline; filename="{stem}.{extension}"'})

    def summary(job: Job) -> JobSummary:
        return JobSummary(id=job.id, filename=job.filename, status=job.status, status_url=f"/jobs/{job.id}",
                          transcript_url=f"/jobs/{job.id}/transcript")

    file_param = File(..., description="WAV, MP3, OGG/Opus, M4A or FLAC audio")
    language_param = Form(None, pattern=LANGUAGE_PATTERN,
                          description="ISO code such as en, de, fr, it; 'auto' to detect")
    format_param = Query(OutputFormat.json, description="json (segments + metadata), text, srt or vtt")

    # ------------------------------------------------------------ synchronous

    @app.post("/transcribe", response_model=TranscriptResponse,
              responses={200: {"content": {"text/plain": {}, "application/x-subrip": {}, "text/vtt": {}}}})
    async def transcribe(request: Request, file: UploadFile = file_param, language: str | None = language_param,
                         format: OutputFormat = format_param):
        """Upload audio and wait for its transcript. Shares the job queue, so it waits its turn."""
        service = service_of(request)
        job = await accept_upload(service, file, language)
        try:
            job = await service.wait(job.id)
        finally:
            if job.id in service.jobs:
                service.delete(job.id)  # the caller has the result; nothing to poll for
        if job.status != JobStatus.completed:
            raise HTTPException(422, f"Transcription failed: {job.error}")
        return render(job, format)

    # ------------------------------------------------------------------- jobs

    @app.post("/jobs", response_model=JobSummary, status_code=202)
    async def create_job(request: Request, file: UploadFile = file_param, language: str | None = language_param):
        """Upload audio for background transcription; poll the status URL, then fetch the transcript."""
        return summary(await accept_upload(service_of(request), file, language))

    @app.get("/jobs", response_model=list[JobSummary])
    async def list_jobs(request: Request):
        return [summary(j) for j in service_of(request).list()]

    @app.get("/jobs/{job_id}", response_model=Job, response_model_exclude={"transcript"})
    async def job_status(request: Request, job_id: str):
        return get_job(service_of(request), job_id)

    @app.get("/jobs/{job_id}/transcript", response_model=TranscriptResponse,
             responses={200: {"content": {"text/plain": {}, "application/x-subrip": {}, "text/vtt": {}}}})
    async def job_transcript(request: Request, job_id: str, format: OutputFormat = format_param):
        job = get_job(service_of(request), job_id)
        if job.status == JobStatus.failed:
            raise HTTPException(422, f"Transcription failed: {job.error}")
        if job.status != JobStatus.completed:
            raise HTTPException(409, f"Transcript not ready (status: {job.status.value})")
        return render(job, format)

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        service = service_of(request)
        get_job(service, job_id)
        service.delete(job_id)

    # ----------------------------------------------------------------- status

    @app.get("/languages")
    async def languages(request: Request):
        """Language codes the loaded model can transcribe."""
        service = service_of(request)
        return {"languages": service.languages, "default": settings.default_language}

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        service = service_of(request)
        return HealthResponse(
            status="ok" if service.transcriber else "degraded",
            model=service.transcriber.name if service.transcriber else None,
            model_error=service.transcriber_error,
            languages=service.languages,
            queue_depth=service.queue.qsize(),
            active_jobs=service.active,
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request):
        """Liveness for the platform health check: fails if the speech model is missing."""
        if service_of(request).transcriber is None:
            raise HTTPException(503, "Speech model not loaded")
        return {"ok": True}

    return app
