"""The training API (app/training.py): try each part of the fine-tuning recipe on the benchmark set's clips, then
start, follow, stop and delete runs. Guarded by API_TOKEN like the rest of the API."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.training import (
    CollatorSettings,
    MetricSettings,
    ProcessorSettings,
    TokenizerSettings,
    TrainingError,
    TrainingManager,
    TrainingRequest,
    TrainingRun,
    TrainingUnavailable,
)
from app.wer import word_error_rate

router = APIRouter(prefix="/training", tags=["training"])


class InspectTokenizer(BaseModel):
    base_model: str | None = None
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    text: str = Field(min_length=1, max_length=2000)


class InspectProcessor(BaseModel):
    base_model: str | None = None
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    processor: ProcessorSettings = Field(default_factory=ProcessorSettings)
    clip_id: str


class InspectCollator(BaseModel):
    base_model: str | None = None
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    processor: ProcessorSettings = Field(default_factory=ProcessorSettings)
    collator: CollatorSettings = Field(default_factory=CollatorSettings)
    clip_ids: list[str] = Field(min_length=1, max_length=8)


class MetricPair(BaseModel):
    reference: str = Field(max_length=5000)
    prediction: str = Field(max_length=5000)


class InspectMetric(BaseModel):
    metric: MetricSettings = Field(default_factory=MetricSettings)
    pairs: list[MetricPair] = Field(min_length=1, max_length=20)


def manager_of(request: Request) -> TrainingManager:
    return request.app.state.training


async def call(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except TrainingUnavailable as e:
        raise HTTPException(503, str(e))
    except TrainingError as e:
        raise HTTPException(422, str(e))
    except KeyError:
        raise HTTPException(404, "No such training run")


@router.get("")
async def overview(request: Request, refresh: bool = False):
    """Whether this service can train (the training Python, its PyTorch and transformers, CPU or GPU), the base models,
    the benchmark set's clips (the data) and every setting's default."""
    manager = manager_of(request)
    info = await asyncio.to_thread(manager.availability, refresh)
    clips = await asyncio.to_thread(manager.clips.list_clips)
    limit = manager.models.limit_mb
    return {**info, "threads": manager.threads, "memory_limit_mb": round(limit) if limit else None,
            "base_models": [{"id": m.id, "default": m.default, "finetuned": m.id.startswith("finetuned/")}
                            for m in manager.models.list()],
            "clips": len(clips), "clip_seconds": round(sum(c.duration_seconds for c in clips), 1),
            "clip_list": [{"id": c.id, "name": c.name, "duration_seconds": c.duration_seconds, "reference": c.reference}
                          for c in clips],  # to try the parts on
            "running": manager.current, "defaults": TrainingRequest().model_dump()}


@router.post("/inspect/tokenizer")
async def inspect_tokenizer(request: Request, body: InspectTokenizer):
    """WhisperTokenizer on a text: the labels' token ids and tokens, decoded with and without the special tokens."""
    manager = manager_of(request)
    base = await call(manager.base, body.base_model)
    return await call(manager.inspect, "tokenizer", {"base": base, "tokenizer": body.tokenizer.model_dump(),
                                                     "text": body.text})


@router.post("/inspect/processor")
async def inspect_processor(request: Request, body: InspectProcessor):
    """WhisperProcessor on a clip: its log-mel spectrogram (shape, a preview) and its labels."""
    manager = manager_of(request)
    base = await call(manager.base, body.base_model)
    clip = (await call(manager.clip_inputs, [body.clip_id]))[0]
    return await call(manager.inspect, "processor", {"base": base, "clip": clip, "tokenizer": body.tokenizer.model_dump(),
                                                     "processor": body.processor.model_dump()})


@router.post("/inspect/collator")
async def inspect_collator(request: Request, body: InspectCollator):
    """The data collator on a batch of clips: the tensors' shapes and each row of labels, padding included."""
    manager = manager_of(request)
    base = await call(manager.base, body.base_model)
    clips = await call(manager.clip_inputs, list(dict.fromkeys(body.clip_ids)))
    return await call(manager.inspect, "collator", {"base": base, "clips": clips,
                                                    "tokenizer": body.tokenizer.model_dump(),
                                                    "processor": body.processor.model_dump(),
                                                    "collator": body.collator.model_dump()})


@router.post("/inspect/metric")
async def inspect_metric(body: InspectMetric):
    """The evaluation metric on pairs of correct text and prediction: computed here, as during training."""
    from app.training_worker import error_rates  # numpy and app.wer only: PyTorch is imported by a run

    refs, preds = [p.reference for p in body.pairs], [p.prediction for p in body.pairs]
    rates = error_rates(refs, preds)
    pairs = []
    for ref, pred in zip(refs, preds):
        result = word_error_rate(ref, pred)
        pairs.append({"reference": ref, "prediction": pred, "wer": round(result.wer, 4), "errors": result.errors,
                      "reference_words": result.reference_words, "alignment": result.alignment})
    return {"metric": body.metric.name, "value": rates[body.metric.name], **rates, "pairs": pairs}


@router.get("/runs", response_model=list[TrainingRun])
async def list_runs(request: Request):
    return await call(manager_of(request).list)


@router.post("/runs", response_model=TrainingRun, status_code=202)
async def start_run(request: Request, body: TrainingRequest):
    """Queue a run: the base model scored on the test clips, trained, scored again, saved and exported. One run at a
    time, never retried; follow it with GET /training/runs/{id}."""
    return await call(manager_of(request).submit, body)


@router.get("/runs/{run_id}", response_model=TrainingRun)
async def get_run(request: Request, run_id: str):
    return await call(manager_of(request).get, run_id)


@router.post("/runs/{run_id}/stop", response_model=TrainingRun)
async def stop_run(request: Request, run_id: str):
    """Stop at the next step; what was learned is still scored, saved and exported. A queued run never starts."""
    return await call(manager_of(request).request_stop, run_id)


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(request: Request, run_id: str):
    """The run, its files and its fine-tuned model (no longer offered to jobs); a running one is killed."""
    await call(manager_of(request).delete, run_id)


@router.get("/runs/{run_id}/log", response_class=PlainTextResponse)
async def run_log(request: Request, run_id: str, lines: int = 200):
    """The end of the training process's output (transformers' messages, a traceback)."""
    manager = manager_of(request)
    await call(manager.get, run_id)
    return await asyncio.to_thread(manager.log_tail, run_id, max(1, min(lines, 2000)))
