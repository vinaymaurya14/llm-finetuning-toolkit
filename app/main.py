"""FastAPI application with all routes for the LLM Fine-Tuning Toolkit."""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    CompareRequest,
    CompareResult,
    DatasetInfo,
    DatasetPrepareRequest,
    EvaluateRequest,
    EvaluationResult,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    GPUInfo,
    ModelInfo,
    SFTTrainingRequest,
    DPOTrainingRequest,
    TrainingJob,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A full-lifecycle LLM fine-tuning toolkit: dataset preparation, "
        "QLoRA/LoRA training (SFT + DPO), evaluation, model registry, and inference."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────── Health ──────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check with GPU info."""
    from app.utils.gpu_utils import detect_gpu
    from app.services.inference import get_loaded_models

    gpu_status = detect_gpu()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        gpu=GPUInfo(
            available=gpu_status.available,
            device_name=gpu_status.device_name,
            vram_total_mb=gpu_status.vram_total_mb,
            vram_used_mb=gpu_status.vram_used_mb,
            vram_free_mb=gpu_status.vram_free_mb,
            cuda_version=gpu_status.cuda_version,
        ),
        models_loaded=len(get_loaded_models()),
    )


# ────────────────────── Datasets ──────────────────────


@app.post("/datasets/prepare", response_model=DatasetInfo, tags=["Datasets"])
async def prepare_dataset(request: DatasetPrepareRequest):
    """Upload/load dataset, convert to training format, validate & split."""
    from app.services.dataset_engine import prepare_dataset as _prepare

    try:
        result = _prepare(
            source=request.source,
            source_format=request.source_format,
            target_format=request.target_format,
            split_ratio=request.split_ratio,
            filters=request.filters,
            name=request.name,
            subset=request.subset,
            max_samples=request.max_samples,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets", response_model=list[DatasetInfo], tags=["Datasets"])
async def list_datasets():
    """List all prepared datasets."""
    from app.services.dataset_engine import list_datasets as _list

    return _list()


# ────────────────────── Training ──────────────────────


@app.post("/training/sft", response_model=TrainingJob, tags=["Training"])
async def start_sft(request: SFTTrainingRequest):
    """Launch SFT (supervised fine-tuning) job with QLoRA."""
    from app.services.trainer import start_sft_training

    try:
        job = await start_sft_training(request)
        return job
    except Exception as e:
        logger.error(f"SFT launch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/dpo", response_model=TrainingJob, tags=["Training"])
async def start_dpo(request: DPOTrainingRequest):
    """Launch DPO (preference alignment) job."""
    from app.services.trainer import start_dpo_training

    try:
        job = await start_dpo_training(request)
        return job
    except Exception as e:
        logger.error(f"DPO launch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status/{job_id}", response_model=TrainingJob, tags=["Training"])
async def get_training_status(job_id: str):
    """Get training job status & metrics."""
    from app.services.trainer import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/training/jobs", response_model=list[TrainingJob], tags=["Training"])
async def list_training_jobs():
    """List all training jobs."""
    from app.services.trainer import list_jobs

    return list_jobs()


# ────────────────────── Evaluation ──────────────────────


@app.post("/evaluate", response_model=EvaluationResult, tags=["Evaluation"])
async def evaluate_model(request: EvaluateRequest):
    """Run evaluation suite on a trained adapter."""
    from app.services.evaluator import evaluate_model as _evaluate

    try:
        result = await _evaluate(request)
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/compare", response_model=CompareResult, tags=["Evaluation"])
async def compare_models(request: CompareRequest):
    """Compare base model vs fine-tuned on benchmark."""
    from app.services.evaluator import compare_models as _compare

    try:
        result = await _compare(
            adapter_path=request.adapter_path,
            base_model=request.base_model,
            dataset_name=request.dataset_name,
            num_samples=request.num_samples,
        )
        return result
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────── Model Registry ──────────────────────


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List registered adapters with metadata."""
    from app.services.model_registry import list_adapters

    return list_adapters()


@app.get("/models/{adapter_name}", response_model=ModelInfo, tags=["Models"])
async def get_model(adapter_name: str):
    """Get a specific adapter by name."""
    from app.services.model_registry import get_adapter

    adapter = get_adapter(adapter_name)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_name}' not found")
    return adapter


@app.delete("/models/{adapter_name}", tags=["Models"])
async def delete_model(adapter_name: str):
    """Remove an adapter from the registry."""
    from app.services.model_registry import delete_adapter

    if delete_adapter(adapter_name):
        return {"message": f"Adapter '{adapter_name}' removed from registry"}
    raise HTTPException(status_code=404, detail=f"Adapter '{adapter_name}' not found")


# ────────────────────── Inference ──────────────────────


@app.post("/inference/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest):
    """Generate text using base model + optional LoRA adapter."""
    from app.services.inference import generate_text as _generate

    try:
        result = await _generate(request)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/unload", tags=["Inference"])
async def unload_models():
    """Unload all cached models to free GPU memory."""
    from app.services.inference import unload_all

    unload_all()
    return {"message": "All models unloaded"}
