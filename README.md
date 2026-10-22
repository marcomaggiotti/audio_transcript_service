# Audio Transcript Service

Send an audio file, get its text back: **Whisper running on ONNX Runtime**, with timestamped
segments, plain text, SRT or WebVTT output.

The speech-recognition code is taken from
[voice_embedding_service](https://github.com/marcomaggiotti/voice_embedding_service) (Whisper ONNX
export, NumPy log-mel features, greedy decoding with Whisper's timestamp rules, audio decoding).
This service keeps only that part. It has no search, embeddings or storage.

```
 POST /transcribe ─────────────► queue ─► ffmpeg decode → 16 kHz mono
 POST /jobs  (202 + job id)              log-mel spectrogram (NumPy)
                                         Whisper encoder / decoder with KV cache (ONNX Runtime)
 GET /jobs/{id}/transcript ◄──────────── timestamped segments → json | text | srt | vtt
```

## What's in it

| Area | Implementation |
|---|---|
| Speech recognition | Whisper exported to ONNX (INT8, KV-cache decoder). Greedy decoding with Whisper's timestamp rules, 30 s windows with seek, language auto-detect (`app/asr.py`) |
| Features | Whisper log-mel spectrogram in pure NumPy, no torch or librosa at runtime (`app/features.py`) |
| Audio | Magic-byte format check, ffmpeg decoding to 16 kHz mono with a pure-Python WAV fallback (`app/audio.py`) |
| Jobs | In-process asyncio queue with bounded concurrency. Synchronous requests use the same queue (`app/jobs.py`) |
| Output | JSON with segments and per-segment confidence, plain text, SRT, WebVTT (`app/formats.py`) |
| Metrics | Per job: decode time, inference time, real-time factor, peak memory |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transcribe?format=json\|text\|srt\|vtt` | Multipart `file` (WAV, MP3, OGG/Opus, M4A, FLAC) and optional `language` (`en`, `de`, `it`, … or `auto`). Waits and returns the transcript |
| `POST` | `/jobs` | Same upload, transcribed in the background. Returns `202` with the job id, `status_url` and `transcript_url` |
| `GET` | `/jobs` | Recent jobs, newest first |
| `GET` | `/jobs/{id}` | Status, `stage` (decoding, transcribing), `progress` (0–1, updated after each 30 s window), `queue_position` while queued, model, timings and metrics, or the error if it failed |
| `GET` | `/jobs/{id}/transcript?format=` | The transcript. `409` until it's ready, `422` if the job failed |
| `DELETE` | `/jobs/{id}` | Forget a job (and its upload if it's still queued) |
| `GET` | `/languages` | Languages the loaded model supports, and the default |
| `GET` | `/health` | Model status, queue depth, active jobs |
| `GET` | `/healthz` | Liveness: `503` if the speech model didn't load |

```bash
# Plain text, waiting for the result
curl -F file=@episode.mp3 "localhost:8000/transcribe?format=text"

# Subtitles in Italian
curl -F file=@clip.m4a -F language=it "localhost:8000/transcribe?format=srt" -o clip.srt

# Long file: queue it, poll, fetch
curl -F file=@talk.mp3 localhost:8000/jobs               # {"id": "…", "status": "queued", …}
curl localhost:8000/jobs/<id>                            # "status": "completed"
curl localhost:8000/jobs/<id>/transcript
```

The JSON transcript looks like this:

```json
{
  "language": "en",
  "duration_seconds": 15.0,
  "text": "Welcome to the show. Today we talk about ONNX Runtime.",
  "segments": [
    {"index": 0, "start": 0.0, "end": 4.2, "text": "Welcome to the show.",
     "avg_logprob": -0.21, "no_speech_prob": 0.01}
  ],
  "model": "openai/whisper-tiny",
  "metrics": {"audio_seconds": 15.0, "decode_seconds": 0.05, "inference_seconds": 2.1,
              "real_time_factor": 0.14, "peak_memory_mb": 310.0}
}
```

`avg_logprob` is the mean log-probability of the segment's tokens: below about −0.7 the text is often
wrong. `no_speech_prob` is Whisper's estimate that the segment's 30 s window is silence.

When `API_TOKEN` is set, `/transcribe`, `/jobs` and `/languages` need `X-API-Key: <token>` or
`Authorization: Bearer <token>`. `/health`, `/healthz` and the API docs (`/docs`) stay public.

## Run locally

```bash
pip install -r requirements-dev.txt
# Once, to export the model. Needs torch (CPU build is enough):
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-export.txt
python scripts/export_whisper_onnx.py --model openai/whisper-tiny --output models/whisper-tiny

WHISPER_MODEL_DIR=models/whisper-tiny python main.py
open http://localhost:8000/docs
```

`ffmpeg` must be on the PATH for MP3, OGG, M4A and FLAC. WAV falls back to a pure-Python decoder.
Without a model directory the service still starts in `degraded` mode: `/health` says why, and
transcription requests get `503`.

Any Whisper checkpoint on Hugging Face works: `openai/whisper-base`, `openai/whisper-small`, or an
English-only `openai/whisper-tiny.en`. Larger models are more accurate but slower and use more RAM.

Tests use a fake Whisper, so they need no model download:

```bash
pytest
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL_DIR` | `./models/whisper-tiny` | Output of `scripts/export_whisper_onnx.py` |
| `DEFAULT_LANGUAGE` | `auto` | Used when a request doesn't specify one; `auto` detects it |
| `MAX_NEW_TOKENS` | `224` | Decoder tokens per 30 s window |
| `ONNX_THREADS` | `0` (auto) | ONNX Runtime threads; set to the instance's CPU count |
| `WHISPER_LOW_MEMORY` | `false` | About 25 MB less RAM, about 1.5x slower transcription |
| `API_TOKEN` | – | Requires a token for the transcription endpoints |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` | `50` / `1800` | Upload limits; decoding stops just after the duration limit |
| `WORKER_CONCURRENCY` / `MAX_QUEUE_SIZE` | `1` / `20` | Parallel transcriptions; uploads get `503` when the queue is full |
| `MAX_FINISHED_JOBS` | `200` | Finished jobs kept for `GET /jobs/{id}`; older ones are forgotten |
| `UPLOAD_DIR` | `./data/uploads` | Uploads wait here until decoded, then are deleted |

## Docker and Render

```bash
docker build -t audio-transcript --build-arg WHISPER_MODEL=openai/whisper-tiny .
docker run -p 10000:10000 -e API_TOKEN=secret audio-transcript
```

The Dockerfile exports Whisper to ONNX in a throwaway build stage (torch never reaches the runtime
image), so the container needs no downloads at boot.

`render.yaml` is a Render Blueprint for the same image: **New → Blueprint**, pick this repository,
then copy the generated `API_TOKEN` from the service's **Environment** tab. The service keeps nothing
on disk, so it needs no persistent disk. To change model size, edit `WHISPER_MODEL` in `render.yaml`
and push; it's passed to the Docker build.

## Benchmark

```bash
python scripts/benchmark.py sample.mp3 --model-dir models/whisper-tiny --runs 3
```

It prints inference latency, real-time factor and peak memory on this machine.

## Limitations

- Job state is in memory. A restart forgets queued and finished jobs. Run one process (the default
  `python main.py`). For several instances, move the queue to Redis (RQ/arq);
  `TranscriptionService.process()` doesn't need to change.
- Decoding is greedy (no beam search, no temperature fallback), as in the original service.
