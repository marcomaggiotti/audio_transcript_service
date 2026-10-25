# Stage 1: export Whisper to ONNX (needs torch; discarded afterwards).
# On Render, the WHISPER_MODEL environment variable is passed in as this build arg.
FROM python:3.12-slim AS export
ARG WHISPER_MODEL=openai/whisper-tiny
# More models a job can choose (POST /jobs model=...), space-separated, e.g. "openai/whisper-small". Only their
# recommended precision pair is exported (encoder fp32, decoder int8), to keep the image smaller.
ARG EXTRA_MODELS=""
# More models for the Benchmark tab to compare, space-separated (e.g. "openai/whisper-base").
ARG BENCHMARK_MODELS=""
WORKDIR /build
COPY requirements-export.txt scripts/export_whisper_onnx.py ./
# --index-url (not --extra-index-url) guarantees the CPU wheel instead of the multi-GB CUDA build.
# torch is pinned to the version the export was verified with (optimum-onnx 0.1.0, transformers 4.57).
# torch declares setuptools as a dependency; install it from PyPI first so the CPU index needn't mirror it.
RUN pip install --no-cache-dir "setuptools>=77.0.3" \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --no-cache-dir -r requirements-export.txt
RUN python export_whisper_onnx.py --model "$WHISPER_MODEL" --output /models/whisper --prune \
 && for model in $EXTRA_MODELS; do \
      python export_whisper_onnx.py --model "$model" --output "/models/$(basename "$model")" --prune --no-variants; \
    done \
 && for model in $BENCHMARK_MODELS; do \
      python export_whisper_onnx.py --model "$model" --output "/models/$(basename "$model")" --prune; \
    done

# Stage 2: runtime with ONNX Runtime only
FROM python:3.12-slim
# ORT_DISABLE_TELEMETRY: onnxruntime 1.29 otherwise keeps resolving its telemetry endpoint and
# writes a device id, even after onnxruntime.disable_telemetry_events().
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=export /models /app/models
COPY app ./app
COPY scripts ./scripts
COPY main.py .

# Models stay under /app/models: a disk mounted at /data would hide whatever the image has there.
ENV WHISPER_MODEL_DIR=/app/models/whisper \
    MODELS_DIR=/app/models \
    BENCHMARK_DIR=/data/benchmark \
    UPLOAD_DIR=/tmp/uploads \
    PORT=10000
EXPOSE 10000
CMD ["python", "main.py"]
