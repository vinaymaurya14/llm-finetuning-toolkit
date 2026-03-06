"""Application configuration with Pydantic settings."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """LLM Fine-Tuning Toolkit configuration."""

    # App
    app_name: str = "LLM Fine-Tuning Toolkit"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model defaults
    base_model: str = "mistralai/Mistral-7B-v0.3"
    max_seq_length: int = 2048

    # QLoRA defaults (optimized for consumer GPU)
    qlora_bits: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Training defaults
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"

    # DPO defaults
    dpo_beta: float = 0.1

    # Paths
    output_dir: str = "./outputs"
    dataset_dir: str = "./datasets"
    registry_file: str = "./outputs/model_registry.json"

    # GPU
    device: str = "auto"

    # Inference
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    # HuggingFace
    hf_token: Optional[str] = None

    model_config = {"env_prefix": "LFT_", "env_file": ".env", "extra": "ignore"}

    @property
    def target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.lora_target_modules.split(",")]

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_path(self) -> Path:
        path = Path(self.dataset_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
