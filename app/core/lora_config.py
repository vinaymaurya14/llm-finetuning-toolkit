"""LoRA/QLoRA configuration builder with presets and validation."""

import logging
from typing import Any, Optional

from app.config import settings
from app.utils.gpu_utils import detect_gpu, estimate_vram_usage

logger = logging.getLogger(__name__)

# LoRA presets
PRESETS = {
    "efficient": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "description": "Low VRAM usage, good for quick experiments",
    },
    "balanced": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "description": "Good balance of quality and efficiency (recommended)",
    },
    "quality": {
        "rank": 64,
        "alpha": 128,
        "dropout": 0.1,
        "description": "Higher quality, requires more VRAM",
    },
}


def build_lora_config(
    preset: str = "balanced",
    rank: Optional[int] = None,
    alpha: Optional[int] = None,
    dropout: Optional[float] = None,
    target_modules: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build LoRA configuration from preset or custom values.

    Returns a dict that can be passed to peft.LoraConfig().
    """
    if preset != "custom" and preset in PRESETS:
        base = PRESETS[preset].copy()
    else:
        base = PRESETS["balanced"].copy()

    config = {
        "r": rank or base["rank"],
        "lora_alpha": alpha or base["alpha"],
        "lora_dropout": dropout if dropout is not None else base["dropout"],
        "target_modules": target_modules or settings.target_modules_list,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    return config


def build_quantization_config(bits: int = 4) -> dict[str, Any]:
    """Build BitsAndBytes quantization configuration.

    Returns a dict that can be passed to BitsAndBytesConfig().
    """
    if bits == 4:
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }
    elif bits == 8:
        return {
            "load_in_8bit": True,
        }
    else:
        return {}


def validate_config_for_gpu(
    model_name: str,
    lora_config: dict[str, Any],
    quantization_bits: int = 4,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> dict[str, Any]:
    """Validate that the config will fit in available GPU memory.

    Returns validation result with recommendations.
    """
    gpu = detect_gpu()

    # Extract model size from name
    import re
    size_match = re.search(r"(\d+\.?\d*)[bB]", model_name)
    model_size = size_match.group(0) if size_match else "7b"

    vram_estimate = estimate_vram_usage(
        model_size_param=model_size,
        quantization_bits=quantization_bits,
        lora_rank=lora_config.get("r", 16),
        batch_size=batch_size,
        seq_length=seq_length,
    )

    result = {
        "valid": True,
        "estimated_vram_gb": vram_estimate["total_estimated_gb"],
        "vram_breakdown": vram_estimate,
        "gpu_available": gpu.available,
        "warnings": [],
    }

    if not gpu.available:
        result["valid"] = False
        result["warnings"].append(
            "No GPU detected. Training requires a CUDA or MPS GPU. "
            "Dataset preparation and evaluation can still run on CPU."
        )
        return result

    if gpu.vram_total_mb:
        available_gb = gpu.vram_total_mb / 1024
        if vram_estimate["total_estimated_gb"] > available_gb * 0.95:
            result["valid"] = False
            result["warnings"].append(
                f"Estimated VRAM ({vram_estimate['total_estimated_gb']:.1f}GB) "
                f"exceeds available ({available_gb:.1f}GB). "
                "Try: lower batch_size, lower lora_rank, or use 4-bit quantization."
            )
        elif vram_estimate["total_estimated_gb"] > available_gb * 0.8:
            result["warnings"].append(
                f"VRAM usage ({vram_estimate['total_estimated_gb']:.1f}GB) is close to "
                f"available ({available_gb:.1f}GB). OOM errors possible."
            )

    return result


def get_presets() -> dict[str, dict]:
    """Return available LoRA presets."""
    return PRESETS
