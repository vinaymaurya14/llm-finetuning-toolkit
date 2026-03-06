"""Request/response Pydantic models for all API endpoints."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Enums ---

class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CONVERSATIONAL = "conversational"
    DPO = "dpo"
    AUTO = "auto"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoRAPreset(str, Enum):
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


# --- Dataset Schemas ---

class QualityFilterConfig(BaseModel):
    min_length: int = Field(default=10, description="Minimum text length in characters")
    max_length: int = Field(default=8192, description="Maximum text length in characters")
    remove_duplicates: bool = Field(default=True, description="Remove exact duplicates")
    near_dedup: bool = Field(default=False, description="Near-duplicate removal via MinHash")
    min_quality_score: float = Field(default=0.0, description="Minimum quality score (0-1)")


class DatasetPrepareRequest(BaseModel):
    source: str = Field(..., description="HuggingFace dataset name, local file path, or URL")
    source_format: DatasetFormat = Field(default=DatasetFormat.AUTO, description="Input format")
    target_format: DatasetFormat = Field(default=DatasetFormat.ALPACA, description="Output format")
    split_ratio: float = Field(default=0.9, ge=0.5, le=0.99, description="Train/val split ratio")
    filters: QualityFilterConfig = Field(default_factory=QualityFilterConfig)
    name: Optional[str] = Field(default=None, description="Custom dataset name")
    subset: Optional[str] = Field(default=None, description="HuggingFace dataset subset")
    max_samples: Optional[int] = Field(default=None, description="Max samples to load")


class DatasetInfo(BaseModel):
    name: str
    num_samples: int
    format: DatasetFormat
    columns: list[str]
    split_sizes: dict[str, int]
    source: str
    created_at: str
    path: str


# --- LoRA Config Schemas ---

class LoRAConfigRequest(BaseModel):
    preset: LoRAPreset = Field(default=LoRAPreset.BALANCED)
    rank: Optional[int] = Field(default=None, ge=4, le=128)
    alpha: Optional[int] = Field(default=None, ge=8, le=256)
    dropout: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    target_modules: Optional[list[str]] = None
    quantization_bits: int = Field(default=4, description="4-bit or 8-bit quantization")


# --- Training Schemas ---

class TrainingArgsRequest(BaseModel):
    learning_rate: Optional[float] = Field(default=None, ge=1e-6, le=1e-2)
    num_epochs: Optional[int] = Field(default=None, ge=1, le=100)
    per_device_batch_size: Optional[int] = Field(default=None, ge=1, le=64)
    gradient_accumulation_steps: Optional[int] = Field(default=None, ge=1, le=64)
    warmup_ratio: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    max_seq_length: Optional[int] = Field(default=None, ge=128, le=8192)
    save_steps: Optional[int] = Field(default=None)
    logging_steps: int = Field(default=10)
    fp16: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)


class SFTTrainingRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of prepared dataset")
    base_model: Optional[str] = Field(default=None, description="HuggingFace model ID")
    lora_config: LoRAConfigRequest = Field(default_factory=LoRAConfigRequest)
    training_args: TrainingArgsRequest = Field(default_factory=TrainingArgsRequest)
    use_unsloth: bool = Field(default=True, description="Use Unsloth for faster training")
    resume_from_checkpoint: Optional[str] = Field(default=None)


class DPOTrainingRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of prepared DPO dataset")
    base_model: Optional[str] = Field(default=None, description="HuggingFace model ID")
    sft_adapter_path: Optional[str] = Field(default=None, description="Path to SFT adapter to start from")
    lora_config: LoRAConfigRequest = Field(default_factory=LoRAConfigRequest)
    training_args: TrainingArgsRequest = Field(default_factory=TrainingArgsRequest)
    beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO beta parameter")


class TrainingMetrics(BaseModel):
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None


class TrainingJob(BaseModel):
    job_id: str
    job_type: str = Field(description="sft or dpo")
    status: TrainingStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    config: dict[str, Any] = Field(default_factory=dict)
    adapter_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# --- Evaluation Schemas ---

class EvaluateRequest(BaseModel):
    adapter_path: str = Field(..., description="Path to LoRA adapter")
    base_model: Optional[str] = Field(default=None)
    dataset_name: Optional[str] = Field(default=None, description="Evaluation dataset")
    metrics: list[str] = Field(default=["perplexity", "bleu", "rouge"], description="Metrics to compute")
    num_samples: int = Field(default=100, ge=1, le=10000)


class EvaluationResult(BaseModel):
    adapter_path: str
    perplexity: Optional[float] = None
    bleu: Optional[float] = None
    rouge: Optional[dict[str, float]] = None
    accuracy: Optional[float] = None
    samples: list[dict[str, str]] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CompareRequest(BaseModel):
    adapter_path: str
    base_model: Optional[str] = None
    dataset_name: Optional[str] = None
    num_samples: int = Field(default=50, ge=1, le=1000)


class CompareResult(BaseModel):
    base_scores: dict[str, float]
    finetuned_scores: dict[str, float]
    improvement_pct: dict[str, float]
    sample_comparisons: list[dict[str, str]] = Field(default_factory=list)


# --- Inference Schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    adapter_path: Optional[str] = Field(default=None, description="Path to LoRA adapter")
    base_model: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int
    generation_time_ms: float
    model: str
    adapter: Optional[str] = None


# --- Model Registry Schemas ---

class ModelInfo(BaseModel):
    adapter_name: str
    base_model: str
    dataset: str
    training_type: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    lora_config: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    file_size_mb: Optional[float] = None
    adapter_path: str


# --- Health ---

class GPUInfo(BaseModel):
    available: bool
    device_name: Optional[str] = None
    vram_total_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None
    cuda_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    gpu: GPUInfo
    models_loaded: int = 0
