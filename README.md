# Audio Transcript Service

Send an audio file, get its text back: **Whisper running on ONNX Runtime**, with timestamped
segments, plain text, SRT or WebVTT output. A web page at `/` has two tabs: **Transcribe**, and
**Benchmark**, which measures the word error rate of models and decoding options on reference
intervals you pick from your own recordings.

The speech-recognition code is taken from
[voice_embedding_service](https://github.com/marcomaggiotti/voice_embedding_service) (Whisper ONNX
export, NumPy log-mel features, greedy decoding with Whisper's timestamp rules, audio decoding).
This service keeps only that part. It has no search or embeddings.

**English only.** The service runs English-only Whisper models (`openai/whisper-*.en`), by default
**`openai/whisper-tiny.en`**: their prompt is just `<|startoftranscript|>`, with no language to choose or detect.
A multilingual export is refused when it loads, and left out of `MODELS_DIR`. Responses still carry
`"language": "en"` (transcripts) and `"languages": ["en"]` (`/health`) for clients that read them.

```
 POST /transcribe ─────────────► queue ─► ffmpeg decode → 16 kHz mono
 POST /jobs  (202 + job id)              log-mel spectrogram (NumPy)
                                         Whisper encoder / decoder with KV cache (ONNX Runtime)
 GET /jobs/{id}/transcript ◄──────────── timestamped segments → json | text | srt | vtt
```

## What's in it

| Area | Implementation |
|---|---|
| Speech recognition | Whisper exported to ONNX (fp32 encoder, INT8 KV-cache decoder: an INT8 encoder garbles real speech, see `scripts/export_whisper_onnx.py`). Whisper's timestamp rules, 30 s windows with seek, temperature fallback, previous-text and initial prompts (`app/asr.py`) |
| Features | Whisper log-mel spectrogram in pure NumPy, no torch or librosa at runtime (`app/features.py`) |
| Audio | Magic-byte format check, ffmpeg decoding to 16 kHz mono with a pure-Python WAV fallback (`app/audio.py`) |
| Jobs | In-process asyncio queue with bounded concurrency. Synchronous requests use the same queue (`app/jobs.py`) |
| Output | JSON with segments and per-segment confidence, plain text, SRT, WebVTT (`app/formats.py`) |
| Metrics | Per job: decode time, inference time, real-time factor, peak memory |
| Benchmark | Reference clips cut from real recordings, word error rate per model and decoding variant, word-level diffs (`app/benchmark.py`, `app/wer.py`, `scripts/wer.py`) |
| Web page | `GET /`: Transcribe and Benchmark tabs (`app/static/index.html`) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transcribe?format=json\|text\|srt\|vtt` | Multipart `file` (WAV, MP3, OGG/Opus, M4A, FLAC) and optional `temperature_fallback`, `condition_on_previous_text`, `reset_on_loop`, `vad` (`true`/`false`), `vad_threshold`, `vad_min_silence_ms`, `beam_size`, `initial_prompt` and `intervals`. Waits and returns the transcript |
| `POST` | `/jobs` | Same upload, transcribed in the background. Returns `202` with the job id, `status_url` and `transcript_url` |
| `GET` | `/jobs` | Recent jobs, newest first |
| `GET` | `/jobs/{id}` | Status, `stage` (decoding, transcribing), `progress` (0–1, updated after each 30 s window), `queue_position` while queued, model, timings and metrics, or the error if it failed |
| `GET` | `/jobs/{id}/transcript?format=` | The transcript. `409` until it's ready, `422` if the job failed |
| `DELETE` | `/jobs/{id}` | Cancel a job: a queued one never runs, a running one stops at its next decoder step (CTranslate2: next segment), so the worker moves on to the next job |
| `GET` | `/config` | How audio becomes text here: decoding, windows, log-mel features (fixed by Whisper) and the model, its encoder/decoder precision (read from the ONNX graphs) and decoding defaults |
| `GET` | `/health` | Model status, queue depth, active jobs |
| `GET` | `/healthz` | Liveness: `503` if the speech model didn't load |
| | `/benchmark/…`, `/benchmarks` | The benchmark: see [below](#benchmark-word-error-rate-on-your-own-audio) |
| | `/training`, `/training/…` | Fine-tuning on the benchmark set: see [below](#fine-tuning-on-the-benchmark-set) |

```bash
# Plain text, waiting for the result
curl -F file=@episode.mp3 "localhost:8000/transcribe?format=text"

# Subtitles
curl -F file=@clip.m4a "localhost:8000/transcribe?format=srt" -o clip.srt

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
  "model": "openai/whisper-tiny.en",
  "metrics": {"audio_seconds": 15.0, "decode_seconds": 0.05, "inference_seconds": 2.1,
              "real_time_factor": 0.14, "peak_memory_mb": 310.0}
}
```

`avg_logprob` is the mean log-probability of the segment's tokens: below about −0.7 the text is often
wrong. `no_speech_prob` is Whisper's estimate that the segment's 30 s window is silence.

## Decoding options

Each 30 s window is decoded greedily (the most likely token each step) unless beam search is asked for. These
options follow openai/whisper:

- **Temperature fallback** (`TEMPERATURE_FALLBACK`, form field `temperature_fallback`): a window whose
  text repeats itself (zlib compression ratio above 2.4) or whose tokens are unsure (mean log-probability
  below −1) is decoded again, sampling at temperatures 0.2, 0.4 … 1.0, until it passes. A window Whisper
  thinks is silence (no-speech probability above 0.6, text unsure) gives no text at all. This fixes most
  hallucinations and repetition loops, at the cost of extra decoding on bad windows only.
- **Previous-text prompt** (`CONDITION_ON_PREVIOUS_TEXT`, `condition_on_previous_text`, **off by default**): each
  window is prompted with the text decoded before it (up to 223 tokens), which keeps names and spelling consistent,
  but a window transcribed wrongly then misleads the next one. This is the setting that reuses earlier words.
  It's dropped after a window that needed a high temperature, and (`RESET_PROMPT_ON_LOOP`, `reset_on_loop`,
  on by default) after a window whose text loops (compression ratio above 2.4), so one mistake doesn't spread
  and a loop doesn't snowball into the next window. The window trace marks such a window `prompt_reset`. With a
  small model such as whisper-tiny.en, turning previous text off altogether is often better: measure it.
- **KV cache** (`KV_CACHE`, `kv_cache`, on by default; Transcribe and Benchmark tabs: "KV cache (faster)"):
  Whisper writes a window one word-piece at a time, each step reading every piece written so far. With the cache,
  the decoder keeps its intermediate results (keys and values) for the pieces already read, so a step only
  computes the new one; without it, every step recomputes them all. That is a speed-up only: either way each step
  sees exactly the same words, right or wrong, and a mistake is not "kept" any more by the cache than without it.
  The transcript is the same (with the int8 decoder, rounding can very rarely change a word: compare both in the
  Benchmark tab). Off, each step also recomputes the attention keys and values of all 1500 encoder frames: about
  10x the decoder work, times the beams with beam search. On Render's 0.5 CPU a 2-minute recording with beam 5 took
  over 12 minutes that way, so keep it on unless you are comparing. The CTranslate2 engine always uses its own cache.
- **Beam search** (`BEAM_SIZE`, `beam_size`, default 1 = greedy, up to 8): at temperature 0 the decoder keeps that
  many candidate texts in parallel and picks the one with the best mean log-probability per token, as openai/whisper
  does with 5. Usually noticeably fewer word errors, more so on a weak model; the decoder is about 3-5x slower (the
  encoder is unchanged). Fallback attempts at higher temperatures still sample one text.
- **KV cache** (`KV_CACHE`, `kv_cache`, on by default): the merged decoder keeps each step's self-attention keys
  and values, and the encoder's cross-attention ones from the first step, so every decoder call feeds only the new
  token. Off, every step re-runs the whole prefix (prompt and text so far) through the decoder: the same text, a
  slower decoder, more so on long windows and with beam search. It's there to measure what the cache saves; the job's
  `metrics.kv_cache` says whether it was used. CTranslate2 always decodes with its cache, and an export with only
  `decoder_model.onnx` (no merged decoder) never has one; `GET /config` (`decoding.kv_cache.available`) says which.
- **Voice activity detection** (`VAD`, `vad`, off by default; `vad_threshold`, `vad_min_silence_ms`): Silero VAD
  (MIT licence, a 2 MB ONNX model at `VAD_MODEL_PATH`, put in the image by the Dockerfile) gives each 32 ms of audio
  a speech probability. Stretches of speech (padded by 200 ms, ended by a pause of at least `vad_min_silence_ms`,
  default 500) are grouped into pieces of up to 29 s, cut at pauses, and only those are transcribed, the silence
  between them left out. Whisper hears only speech, which removes most invented phrases ("Thanks for watching")
  and many loops, since those start in silence and noise; no word is split at a fixed window edge; and silence
  costs no decoding time. Segment times are mapped back to the recording, and the transcript lists the stretches
  of speech (`speech`). Detection takes about 1 s per 4 minutes of audio. Without the model file a request with
  `vad=true` is refused (503); `GET /config` says whether VAD is available.
- **Initial prompt** (`INITIAL_PROMPT`, `initial_prompt`): words and spellings the audio contains, such as
  names and product terms, given to every window.

Sampling uses a fixed seed, so the same audio gives the same transcript. Whether these options help
depends on the model and the audio: measure it with the benchmark.

### The decoding loop, window by window

The JSON transcript also has `windows`: one entry per 30 s window, in order, with where it started (`start`, the
seek position), how much audio it held (`length`), how far the seek moved on (`consumed`: to the last complete
timestamp, at least 1 s), Whisper's `no_speech_prob`, whether its text was dropped as silence (`skipped`), how many
segments it gave, and every decode (`attempts`: temperature, compression ratio, mean log-probability; with the
temperature fallback a window can be decoded up to six times, and the last attempt is the one kept).

### Transcribing on given intervals

`POST /transcribe` and `POST /jobs` take an optional `intervals` field: a JSON list of `[start, end]` pairs in seconds,
in order and not overlapping, e.g. `[[0, 12.4], [12.4, 31]]`. Each span is then transcribed on its own, with no
text carried from one span to the next (`condition_on_previous_text` is off for them), and the transcript has exactly
one segment per span, at the span's times: its text is everything Whisper heard in it, empty when it heard nothing.
This is for transcribing an edited copy of a recording (cleaned, equalized) on the original's lines, so the two
transcripts line up row by row and no words spill from one line into the next. `windows` then lists the windows of
every span, at their place in the whole audio.

## Benchmark: word error rate on your own audio

Speed tells you little about which model to use; accuracy on *your* audio does. The **Benchmark** tab
(`GET /`) scores models and decoding variants by word error rate (WER):

1. **Pick reference intervals.** Upload a real recording. Whisper transcribes it and shows its
   segments on a timeline, colour-coded by confidence. Select a segment, or shift-click to span several,
   adjust start and end to the tenth of a second (or set them from the player's position), play the
   interval, and type what is *really* said. Save it as a reference clip. Choose intervals that are
   representative: clear, complete sentences, from different speakers and conditions. About 20–30 clips
   give a stable score. You can also import clips you cut yourself, with their transcripts.
2. **Reference clips** lists the set. Transcripts can be edited in place; unticked clips are skipped.
3. **Run** with any models and decoding variants (greedy, temperature fallback, fallback + previous
   text) and an optional initial prompt. Runs go through the transcription queue, so they
   never run at the same time as a job.
4. **Results** shows WER, substitutions, deletions and insertions, real-time factor, peak memory and
   load time for each combination, with the lowest WER highlighted. Click a row for every clip's
   word-level differences against its reference.

WER = (wrong + missing + extra words) / words in the references, after lower-casing and removing
punctuation and bracketed notes like `[music]`.

**Models to compare** come from `MODELS_DIR` (default `./models`): every subdirectory holding an export.
Export the candidates next to each other:

```bash
python scripts/export_whisper_onnx.py --model openai/whisper-tiny.en --output models/whisper-tiny.en
python scripts/export_whisper_onnx.py --model openai/whisper-base.en --output models/whisper-base.en
```

The service's own model stays loaded; another model is loaded for its part of a run and freed
afterwards, so a run needs room for both (whisper-base.en next to tiny.en: about +150 MB).

**On disk** (`BENCHMARK_DIR`, default `./data/benchmark`): recordings in `sources/`, reference clips in
`clips/` as `<id>.wav` + `<id>.txt` (the correct transcript) + `<id>.json`, and finished runs in `runs/`.

**From the command line**, the same scoring works on any folder of audio files with same-name `.txt`
references, including the Benchmark tab's `clips/` folder:

```bash
python scripts/wer.py data/benchmark/clips --model-dir models/whisper-tiny.en --model-dir models/whisper-base.en \
    --details
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/benchmark/sources` | Upload a recording (`file`); Whisper proposes its segments |
| `GET` | `/benchmark/sources`, `/benchmark/sources/{id}` | Recordings; one with its proposed segments |
| `GET` | `/benchmark/sources/{id}/audio` | The recording |
| `POST` | `/benchmark/sources/{id}/transcribe` | Propose segments again, e.g. after changing the decoding defaults |
| `DELETE` | `/benchmark/sources/{id}` | Delete a recording (its clips stay) |
| `POST` | `/benchmark/clips` | `{"source_id", "start", "end", "reference", "name"}`: cut an interval into a reference clip |
| `POST` | `/benchmark/clips/upload` | Multipart `file`, `reference`, `name`: a clip you cut yourself |
| `GET` / `PATCH` / `DELETE` | `/benchmark/clips[/{id}]` | List, edit the transcript or name, delete |
| `GET` | `/benchmark/clips/{id}/audio` | The clip as 16 kHz WAV |
| `GET` | `/benchmark/models`, `/benchmark/variants` | What a run can compare |
| `POST` | `/benchmarks` | `{"models", "variants", "initial_prompt", "clip_ids", "kv_cache"}`: queue a run (`202`) |
| `GET` | `/benchmarks`, `/benchmarks/{id}` | Runs with progress and results; one run with per-clip diffs |
| `DELETE` | `/benchmarks/{id}` | Delete a finished run |

When `API_TOKEN` is set, `/transcribe`, `/jobs`, `/benchmark/…` and `/benchmarks` need
`X-API-Key: <token>` or `Authorization: Bearer <token>`. The web page asks for the token once (top
right). The page itself, `/health`, `/healthz` and the API docs (`/docs`) stay public.

### Encoder and decoder precision

The image keeps both precisions of each part of the model: the encoder in fp32 (the default) and int8, the
decoder in int8 (the default) and fp32. `POST /jobs` and `POST /transcribe` take `encoder=fp32|int8` and
`decoder=fp32|int8`; the job and its transcript record the pair used (`precision`). The default pair stays
loaded; a job that asks for another part swaps it in (the default one is released first, the requested one
loaded in under a second for tiny) and the default is loaded back afterwards, so two versions of a part are
never in memory together. Measured on whisper-tiny, one thread, low-memory mode, a 6 s voice note:

| Encoder / decoder | Peak RSS | Time | Note |
|---|---|---|---|
| fp32 / int8 (default) | 290 MB | 0.6 s | |
| fp32 / fp32 | 312 MB | 1.5 s | most accurate (WER 5.0% vs 5.7% on the test set) |
| int8 / int8 | 209 MB | 0.9 s | garbles real speech |
| int8 / fp32 | 288 MB | 1.3 s | garbles real speech |

`GET /config` lists what the export has (`model.available_precisions`). `--no-variants` on the export keeps
only the default pair (about 120 MB smaller image; jobs can't choose).

**INT4.** The export also keeps 4-bit copies of both parts (`encoder_model.int4.onnx`,
`decoder_model_merged.int4.onnx`): every MatMul with a constant weight becomes ONNX Runtime's `MatMulNBits`, weights
in blocks of 32 with a scale each, activations still fp32: about a quarter of the fp32 files, for memory-constrained
devices. `encoder=int4` and `decoder=int4` pick them (`--no-int4` on the export skips them). Expect more word errors
than int8: compare them on your audio (Aipollo's Benchmark tab, Testbench).

### Engine per job: ONNX Runtime or CTranslate2

The same checkpoint is also converted for CTranslate2 at build time (`--ct2` on the export: `models/whisper/ct2`,
float16 weights; the `CTRANSLATE2=false` build arg skips it). `POST /jobs` and `POST /transcribe` take
`engine=ctranslate2` and `compute_type=int8|int8_float32|float32` (default `int8`, CTranslate2's usual CPU choice):
the job runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with its own windowing, beam search,
temperature fallback and VAD (its bundled Silero), from the job's decoding options. The model is loaded for the job
and released after it, like another model; CTranslate2 runs the service's own model only (`model` is refused), takes
`compute_type` instead of `encoder`/`decoder`, and is checked against the memory left first. `GET /config` reports
both engines (`engines.onnx`, `engines.ctranslate2`: available, error, compute types, size).

Every job's metrics now say which engine ran (`metrics.engine`), the size of the weights it ran with
(`metrics.model_mb`: the files of the precisions used), and its own peak memory (`metrics.peak_memory_mb`: the peak
is reset when the job starts, on Linux), so jobs can be compared one by one.

### Model per job

Besides its own model (`WHISPER_MODEL`, loaded at start), the image can hold more English-only ones
(`EXTRA_MODELS`, e.g. `openai/whisper-base.en`, exported with only the recommended precision pair). `POST /jobs` and
`POST /transcribe` take `model=openai/whisper-base.en` (or `whisper-base.en`); the job records the model used
(`model`). Another model
is loaded for the job and released after it, like a non-default precision, so at most one extra model is in
memory; its load time is reported apart (`metrics.load_seconds`).

Before queuing, the service checks the model exists, has the precisions asked for, and fits in the container's
memory: its files × 1.3 + 150 MB, next
to what the process already holds, within 90% of the cgroup limit (`MEMORY_LIMIT_MB` overrides it). A model
that doesn't fit is refused with `422` and the reason, instead of the process being killed. `GET /config`
lists the models (`models`: precisions, `needs_mb`, `fits`, `reason`) and `memory_limit_mb`.

| Model | Size | Memory while transcribing | Render plan |
|---|---|---|---|
| whisper-tiny.en (default) | 39 M parameters | ~300 MB | starter (512 MB) |
| whisper-base.en | 74 M parameters | between the two (not measured yet) | standard (2 GB) |
| whisper-small.en | 244 M parameters | ~1 GB, next to tiny.en | standard (2 GB) |

## Fine-tuning on the benchmark set

The benchmark set's clips (lines of your recordings with their correct text) can also teach the model: `/training`
fine-tunes it with Hugging Face's recipe (huggingface.co/blog/fine-tune-whisper), and the podcast app's **Training**
tab shows each part with its settings:

- **WhisperTokenizer**: a clip's correct text → the token ids the model learns to write (the labels). The models
  here are English-only: the labels start `<|startoftranscript|><|notimestamps|>`, with no language or task token.
  Settings: `predict_timestamps`, `max_label_tokens` (448 at most), `text` (`as_written` keeps case and punctuation,
  what Whisper writes; `normalized` is lower case without punctuation). `POST /training/inspect/tokenizer` shows the
  tokens of a text, decoded with and without the special tokens.
- **WhisperFeatureExtractor** and **WhisperProcessor**: 16 kHz audio → an 80 × 3000 log-mel spectrogram (30 s,
  padded). Clips longer than 30 s are left out. `POST /training/inspect/processor` shows a clip's spectrogram.
- **Data collator** (`DataCollatorSpeechSeq2SeqWithPadding`): pads each batch's labels with −100 (ignored by the
  loss) and cuts `<|startoftranscript|>` (the model adds it). `POST /training/inspect/collator` shows a batch.
- **Evaluation metrics**: word error rate with the benchmark's normalization (`app/wer.py`), or character error rate;
  the chosen one picks the best checkpoint. `POST /training/inspect/metric` scores pairs of texts.
- **Training**: `Seq2SeqTrainer` with `Seq2SeqTrainingArguments` (`learning_rate`, `per_device_train_batch_size`,
  `gradient_accumulation_steps`, `warmup_steps`, `max_steps`, `eval_steps`, `logging_steps`, `generation_max_length`,
  `lr_scheduler_type`, `weight_decay`, `gradient_checkpointing`, `fp16` on a GPU, `load_best_model_at_end`, `seed`),
  plus `freeze_encoder` (on by default: only the decoder, which writes the words, learns; less memory and less
  overfitting on a small set).

`POST /training/runs` queues a run (one at a time, never retried): the clips are shuffled and split (`test_size`,
`seed`), the base model is scored on the test clips, trained (evaluated every `eval_steps` with `generate()`),
scored again with its best checkpoint, saved and exported to ONNX (`scripts/export_whisper_onnx.py`) into
`TRAINING_DIR/models/<name>`. It then joins the models a job can choose (`model=finetuned/<name>`) and the benchmark
can compare; its PyTorch weights stay next to it, so it can be trained further. `GET /training/runs/{id}` has the
progress, the loss and evaluation history, the error rates before and after with examples, and the clips left out;
`POST /training/runs/{id}/stop` stops at the next step (what it learned is still scored and exported);
`DELETE` removes the run and its model; `GET /training/runs/{id}/log` is the end of the training process's output.
Score a fine-tuned model on its test clips only: it has seen the others.

Training never runs in the API's Python. It runs in `TRAINING_PYTHON`, which needs PyTorch and transformers: the
Docker image installs them in `/opt/train` with `--build-arg TRAINING=true` (+1 GB; transformers 4.57 needs an older
huggingface-hub than faster-whisper, hence a Python of its own), and keeps `WHISPER_MODEL`'s PyTorch weights to
start from. Without it, `GET /training` says what is missing. On a CPU, whisper-tiny.en trains a few steps a minute
and needs about 2–3 GB of memory (Render pro, not starter); use a GPU machine (`TRAINING_PYTHON` pointing at a
Python with CUDA PyTorch) for larger models or sets. About 50 clips or more make a difference.

Locally:

```bash
python -m venv .train && .train/bin/pip install torch -r requirements-export.txt
TRAINING_PYTHON=.train/bin/python python main.py
.train/bin/python -m pytest tests/test_training_worker.py   # a tiny random Whisper: trains, exports, transcribes
```

## Run locally

```bash
pip install -r requirements-dev.txt
# Once, to export the model. Needs torch (CPU build is enough):
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-export.txt
python scripts/export_whisper_onnx.py --model openai/whisper-tiny.en --output models/whisper-tiny.en

WHISPER_MODEL_DIR=models/whisper-tiny.en python main.py
open http://localhost:8000          # web page; API docs at /docs
```

`ffmpeg` must be on the PATH for MP3, OGG, M4A and FLAC. WAV falls back to a pure-Python decoder.
Without a model directory the service still starts in `degraded` mode: `/health` says why, and
transcription requests get `503`.

Any English-only Whisper checkpoint on Hugging Face works: `openai/whisper-tiny.en`, `openai/whisper-base.en`,
`openai/whisper-small.en` or `openai/whisper-medium.en`. Larger models are more accurate but slower and use more
RAM. A multilingual one (`openai/whisper-tiny`, ...) is refused when it loads.

Tests use a fake Whisper, so they need no model download:

```bash
pytest
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL_DIR` | `./models/whisper-tiny.en` | Output of `scripts/export_whisper_onnx.py` |
| `WHISPER_MODEL` | `openai/whisper-tiny.en` | Build arg (Docker, Render): the model exported as the service's own |
| `MAX_NEW_TOKENS` | `224` | Decoder tokens per 30 s window |
| `ONNX_THREADS` | `0` (auto) | ONNX Runtime threads; set to the instance's CPU count |
| `WHISPER_LOW_MEMORY` | `false` | About 25 MB less RAM, about 1.5x slower transcription |
| `TEMPERATURE_FALLBACK` / `CONDITION_ON_PREVIOUS_TEXT` | `true` / `false` | Decoding defaults; requests can override them |
| `KV_CACHE` | `true` | Decoder KV cache: same transcript, about 10x less decoder work; requests can override it |
| `BEAM_SIZE` / `RESET_PROMPT_ON_LOOP` | `1` / `true` | Beam search width (1 = greedy); stop carrying text after a loop |
| `VAD` / `VAD_THRESHOLD` / `VAD_MIN_SILENCE_MS` | `false` / `0.5` / `500` | Voice activity detection defaults; requests can override them |
| `KV_CACHE` | `true` | The decoder's KV cache (one new token per step); `false` re-runs the whole prefix every step. Requests can override it (`kv_cache`) |
| `VAD_MODEL_PATH` | `./models/vad/silero_vad.onnx` | Silero VAD model (the Docker image has it) |
| `INITIAL_PROMPT` | – | Default initial prompt (names, terms) |
| `MODELS_DIR` | `./models` | Exports jobs can choose (`model=`) and the benchmark can compare |
| `EXTRA_MODELS` | – | Build arg: more English-only models exported into the image for jobs, e.g. `openai/whisper-base.en` |
| `MEMORY_LIMIT_MB` | the cgroup limit | Memory the "does this model fit" check plans with |
| `BENCHMARK_DIR` | `./data/benchmark` | Recordings, reference clips and runs |
| `MAX_CLIP_SECONDS` | `300` | Longest reference clip |
| `API_TOKEN` | – | Requires a token for the transcription endpoints |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` | `50` / `1800` | Upload limits; decoding stops just after the duration limit |
| `WORKER_CONCURRENCY` / `MAX_QUEUE_SIZE` | `1` / `20` | Parallel transcriptions; uploads get `503` when the queue is full |
| `MAX_FINISHED_JOBS` | `200` | Finished jobs kept for `GET /jobs/{id}`; older ones are forgotten |
| `UPLOAD_DIR` | `./data/uploads` | Uploads wait here until decoded, then are deleted |
| `TRAINING` | `false` | Build arg: PyTorch and transformers in the image (`/opt/train`) for fine-tuning |
| `TRAINING_PYTHON` | this Python (`/opt/train/bin/python` in Docker) | The Python that fine-tunes: needs torch and transformers |
| `TRAINING_DIR` | `./data/training` | Training runs, their logs and the fine-tuned models |
| `TRAINING_THREADS` | `0` (PyTorch decides) | PyTorch's threads while training |
| `TRAINING_WORK_DIR` | the system's temp dir | Intermediate checkpoints (deleted after the run) |

## Docker and Render

```bash
docker build -t audio-transcript .                    # whisper-tiny.en
docker build -t audio-transcript --build-arg WHISPER_MODEL=openai/whisper-base.en .
docker run -p 10000:10000 -e API_TOKEN=secret audio-transcript
```

The Dockerfile exports Whisper to ONNX in a throwaway build stage (torch never reaches the runtime
image), so the container needs no downloads at boot. To compare other models in the Benchmark tab,
export them into the image too: `--build-arg BENCHMARK_MODELS="openai/whisper-base.en openai/whisper-small.en"`.
Mount a volume at `/data` to keep the benchmark's recordings, clips and runs, and the fine-tuned models
(`--build-arg TRAINING=true`: see Fine-tuning).

`render.yaml` is a Render Blueprint for the same image: **New → Blueprint**, pick this repository,
then copy the generated `API_TOKEN` from the service's **Environment** tab. A 1 GB disk at `/data` keeps
the benchmark set across deploys. To change model size, edit `WHISPER_MODEL` in `render.yaml` and push;
it's passed to the Docker build, like `BENCHMARK_MODELS`. On the 512 MB starter plan, comparing a second
model next to whisper-tiny.en may run out of memory: use `standard` for benchmarks with whisper-base.en or larger.

## Speed benchmark

```bash
python scripts/benchmark.py sample.mp3 --model-dir models/whisper-tiny.en --runs 3
```

It prints inference latency, real-time factor and peak memory on this machine.

## Limitations

- Job state is in memory. A restart forgets queued and finished jobs. Run one process (the default
  `python main.py`). For several instances, move the queue to Redis (RQ/arq);
  `TranscriptionService.process()` doesn't need to change.
- No beam search: each attempt is greedy or sampled, as in openai/whisper's default settings.
- A benchmark run interrupted by a restart is marked failed; start it again.
