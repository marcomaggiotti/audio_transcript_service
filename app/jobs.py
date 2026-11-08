"""Background transcription jobs: decode -> Whisper -> timestamped transcript.

Jobs run on an in-process asyncio queue; the CPU-bound work runs in worker threads and
`concurrency` bounds how many inferences run at once (Whisper's memory peak is per job).
Synchronous requests and benchmark runs go through the same queue, so they are bounded the same way.
Job state lives in memory: a restart forgets it, and the uploads waiting in the queue are removed.
"""

import asyncio
import functools
import logging
import time
from collections import OrderedDict
from contextlib import nullcontext
from collections.abc import Callable
from pathlib import Path

from app.asr import Cancelled, DecodingOptions, Transcriber, cancel_check
from app.audio import InvalidAudioError, decode_audio, duration_seconds
from app.ct2 import CT2Engine
from app.metrics import peak_memory_mb, reset_peak_memory
from app.models import ModelSet
from app.schemas import InferenceMetrics, Job, JobStatus, Transcript, TranscriptSegment, WindowTrace, utcnow
from app.vad import SileroVad, transcribe_with

logger = logging.getLogger(__name__)


def _cache_counts(windows) -> dict:
    """How many of a job's windows were encoded, and how many of those came from the encoder cache (ONNX only)."""
    if not windows:
        return {}
    return {"windows": len(windows),
            "encoder_cache_hits": sum(1 for w in windows if getattr(w, "encoder_cached", False))}


class QueueFullError(RuntimeError):
    pass


class JobNotFound(KeyError):
    pass


class TranscriptionService:
    def __init__(self, transcriber: Transcriber | None, upload_dir: str, transcriber_error: str | None = None,
                 max_audio_seconds: int = 1800, concurrency: int = 1, max_queue_size: int = 20,
                 max_finished_jobs: int = 200, models: ModelSet | None = None, vad: SileroVad | None = None,
                 ct2: CT2Engine | None = None):
        self.transcriber = transcriber
        self.ct2 = ct2  # the same model on CTranslate2, for jobs with engine=ctranslate2
        self.vad = vad  # voice activity detection, for jobs that ask for it
        self.vad_error: str | None = None  # why it isn't there
        # The models a job can choose; by default only the service's own.
        self.models = models or ModelSet(transcriber, None, None, None)
        self.transcriber_error = transcriber_error
        self.upload_dir = Path(upload_dir)
        self.max_audio_seconds = max_audio_seconds
        self.concurrency = concurrency
        self.max_finished_jobs = max_finished_jobs
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self._finished: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, Callable[[], None]] = {}  # queued work that isn't a transcription job
        self._cancelled: set[str] = set()  # running jobs deleted: they stop at their next decoder step
        self._workers: list[asyncio.Task] = []
        self.active = 0  # jobs currently being transcribed

    def upload_path(self, job_id: str) -> Path:
        return self.upload_dir / job_id

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Jobs are in memory only, so uploads left by a previous process can never be picked up.
        for leftover in self.upload_dir.iterdir():
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    # ------------------------------------------------------------------ jobs

    def submit(self, job: Job) -> None:
        """Queue a job whose audio is already at upload_path(job.id).

        On a full queue the upload is removed and QueueFullError is raised.
        """
        self.jobs[job.id] = job
        self._finished[job.id] = asyncio.Event()
        try:
            self.queue.put_nowait(job.id)
        except asyncio.QueueFull as e:
            self.delete(job.id)
            raise QueueFullError("Transcription queue is full, try again later") from e

    def get(self, job_id: str) -> Job:
        try:
            job = self.jobs[job_id]
        except KeyError:
            raise JobNotFound(job_id) from None
        if job.status == JobStatus.queued:  # jobs are kept in submission order
            ahead = 0
            for other in self.jobs.values():
                if other.id == job_id:
                    break
                ahead += other.status == JobStatus.queued
            job.queue_position = ahead
        else:
            job.queue_position = None
        return job

    def list(self) -> list[Job]:
        return list(reversed(self.jobs.values()))

    def delete(self, job_id: str) -> None:
        """Forget a job. A queued one is skipped; a running one stops at its next decoder step (CTranslate2: its next
        segment), so the worker moves on to the next job instead of finishing work nobody will read."""
        job = self.get(job_id)
        if job.status == JobStatus.processing:
            self._cancelled.add(job_id)
        del self.jobs[job_id]
        self._finished.pop(job_id, None)
        self.upload_path(job_id).unlink(missing_ok=True)

    def submit_task(self, task_id: str, run: Callable[[], None]) -> None:
        """Queue other inference work (a benchmark run) behind the jobs, on the same workers."""
        self._tasks[task_id] = run
        try:
            self.queue.put_nowait(task_id)
        except asyncio.QueueFull as e:
            del self._tasks[task_id]
            raise QueueFullError("Transcription queue is full, try again later") from e

    async def wait(self, job_id: str) -> Job:
        """Wait until the job is completed or failed and return it."""
        job = self.get(job_id)
        event = self._finished.get(job_id)
        if event is not None:
            await event.wait()
        return job

    async def drain(self) -> None:
        """Wait until every queued job has finished (used by tests and benchmarks)."""
        await self.queue.join()

    async def _worker(self, worker_id: int) -> None:
        while True:
            job_id = await self.queue.get()
            task = self._tasks.pop(job_id, None)
            try:
                self.active += 1
                try:
                    await asyncio.to_thread(task or functools.partial(self.process, job_id))
                finally:
                    self.active -= 1
            except Exception:  # process() records failures itself; never kill the worker
                logger.exception("Worker %d crashed on %s", worker_id, job_id)
            finally:
                if (event := self._finished.pop(job_id, None)) is not None:
                    event.set()
                self._prune()
                self.queue.task_done()

    def _prune(self) -> None:
        finished = [j.id for j in self.jobs.values() if j.status in (JobStatus.completed, JobStatus.failed)]
        for job_id in finished[: max(len(finished) - self.max_finished_jobs, 0)]:
            self.jobs.pop(job_id, None)

    # ------------------------------------------------------------------- job

    def process(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return  # deleted while queued
        job.status, job.started_at, job.stage = JobStatus.processing, utcnow(), "decoding"
        path = self.upload_path(job_id)
        reset_peak_memory()  # the peak reported is this job's (Linux), not the largest one so far

        def check() -> None:
            if job_id in self._cancelled:
                raise Cancelled(f"Job {job_id} was cancelled")

        token = cancel_check.set(check)  # this worker thread only: checked before every decoder step
        try:
            self._process(job, path, check)
        except Cancelled:
            logger.info("Stopped cancelled job %s", job_id)
            path.unlink(missing_ok=True)
        finally:
            cancel_check.reset(token)
            self._cancelled.discard(job_id)

    def _process(self, job: Job, path, check: Callable[[], None]) -> None:
        job_id = job.id
        try:
            if self.transcriber is None:
                raise RuntimeError(f"Speech model unavailable: {self.transcriber_error or 'not loaded'}")

            t0 = time.perf_counter()
            audio = decode_audio(path, self.max_audio_seconds, as_int16=True)
            path.unlink(missing_ok=True)  # decoded: the upload isn't needed any more
            seconds = duration_seconds(audio)
            if seconds > self.max_audio_seconds:
                raise InvalidAudioError(f"Audio is longer than the {self.max_audio_seconds}s limit")
            t1 = time.perf_counter()
            job.stage = "transcribing"

            def progress(share: float) -> None:
                check()  # also stops the CTranslate2 engine, between segments
                job.progress = round(share, 3)

            options = DecodingOptions(temperature_fallback=job.temperature_fallback,
                                      condition_on_previous_text=job.condition_on_previous_text,
                                      initial_prompt=job.initial_prompt, beam_size=job.beam_size,
                                      reset_on_loop=job.reset_on_loop, vad=job.vad, vad_threshold=job.vad_threshold,
                                      vad_min_silence_ms=job.vad_min_silence_ms, kv_cache=job.kv_cache)
            def loading(model_id: str) -> None:
                job.stage = f"loading {model_id}"

            if job.engine == "ctranslate2":  # the same model on CTranslate2, loaded for the job and released after
                if self.ct2 is None:
                    raise RuntimeError("CTranslate2 is unavailable on this service")
                job.stage = f"loading {self.ct2.name} on CTranslate2"
                loaded = [self.ct2.load(job.compute_type)]
                t_load = time.perf_counter()  # inference is timed from here: loading isn't inference
                job.stage, job.model = "transcribing", self.ct2.name
                job.precision = {"encoder": job.compute_type or "int8", "decoder": job.compute_type or "int8"}
                weights_mb = self.ct2.size_mb()
                result = self.ct2.transcribe(loaded.pop(), audio, options, progress)
                extra = {"kv_cache": True}  # CTranslate2 always decodes with its KV cache
            else:
                result, t_load, weights_mb, extra = self._onnx(job, audio, options, progress, loading)
            t2 = time.perf_counter()
        except Cancelled:
            raise
        except Exception as e:
            logger.exception("Transcription failed for %s", job_id)
            job.error = str(e) or e.__class__.__name__
            job.status, job.finished_at, job.stage = JobStatus.failed, utcnow(), None
            path.unlink(missing_ok=True)
            return
        self._finish(job, result, seconds, t0, t1, t_load, t2, weights_mb, extra)

    def _onnx(self, job: Job, audio, options: DecodingOptions, progress, loading):
        """ONNX Runtime: the service's model, or the one the job asked for, in the precisions and at the graph
        optimization level asked for. Returns the result, when inference started (after any loading or reopening), the
        size of the weights it ran with, and metrics: whether the decoder used its KV cache, the graph optimization
        level and how long opening the sessions at it took."""
        # The service's model, or the one the job asked for, loaded for it and released after it.
        with self.models.use(job.requested_model, on_load=loading) as transcriber:
            job.stage = "transcribing"
            # Only the speech (VAD), and one segment per requested interval, when the job asks for them.
            def run(audio, on_progress, options, _t=transcriber):
                return transcribe_with(_t, audio, on_progress, options, self.vad, job.intervals)
            job.model = transcriber.name
            switch = getattr(transcriber, "precision_for", None)  # absent on test fakes
            optimize = getattr(transcriber, "optimization_for", None)
            graph = {"level": None, "session_seconds": None}
            if switch is None:
                job.precision = getattr(transcriber, "precision", None)
                t_load = time.perf_counter()  # inference is timed from here: loading isn't inference
                result = run(audio, on_progress=progress, options=options)
            else:
                # The graph optimization level first (it reopens the default precisions), then the precisions (opened
                # at that level); inference is timed once both are in place.
                with (optimize(job.graph_optimization) if optimize else nullcontext(graph)) as graph, \
                        switch(job.encoder_precision, job.decoder_precision) as used:
                    job.precision = used
                    t_load = time.perf_counter()
                    result = run(audio, on_progress=progress, options=options)
            weights = getattr(transcriber, "weights_mb", lambda used: None)(job.precision)
            available = getattr(transcriber, "kv_cache_available", None)  # None: a test fake, can't say
        return result, t_load, weights, {"kv_cache": None if available is None else bool(available and options.kv_cache),
                                         "graph_optimization": graph["level"], "session_seconds": graph["session_seconds"]}

    def _finish(self, job: Job, result, seconds: float, t0: float, t1: float, t_load: float, t2: float,
                weights_mb: float | None, extra: dict | None = None) -> None:
        job.transcript = Transcript(
            duration_seconds=round(seconds, 2),
            text=result.text,
            segments=[TranscriptSegment(index=i, start=s.start, end=s.end, text=s.text,
                                        avg_logprob=getattr(s, "avg_logprob", None),
                                        no_speech_prob=getattr(s, "no_speech_prob", None))
                      for i, s in enumerate(result.segments)],
            windows=[WindowTrace(start=w.start, length=w.length, consumed=w.consumed, no_speech_prob=w.no_speech_prob,
                                 skipped=w.skipped, segments=w.segments, attempts=w.attempts,
                                 prompt_reset=getattr(w, "prompt_reset", False),
                                 encoder_cached=getattr(w, "encoder_cached", False))
                     for w in getattr(result, "windows", None) or []],
            speech=getattr(result, "speech", None),
        )
        job.metrics = InferenceMetrics(
            audio_seconds=round(seconds, 2),
            decode_seconds=round(t1 - t0, 3),
            inference_seconds=round(t2 - t_load, 3),
            real_time_factor=round((t2 - t_load) / seconds, 3) if seconds else 0.0,
            load_seconds=round(t_load - t1, 3) if t_load - t1 >= 0.01 else None,
            peak_memory_mb=peak_memory_mb(),
            engine=job.engine, model_mb=weights_mb, **(extra or {}),
            **_cache_counts(getattr(result, "windows", None)),
        )
        # Status last, so a reader that sees "completed" also sees the transcript.
        job.progress, job.stage = 1.0, None
        job.status, job.finished_at = JobStatus.completed, utcnow()
        logger.info("Transcribed %s with %s (%s): %.1fs audio in %.2fs", job.id, job.model, job.engine, seconds, t2 - t_load)
