"""GPU detection, VRAM estimation, and optimization utilities."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model size estimates in GB (approximate, in float16)
MODEL_SIZES_GB = {
    "1b": 2.0,
    "3b": 6.0,
    "7b": 14.0,
    "8b": 16.0,
    "13b": 26.0,
    "70b": 140.0,
}


@dataclass
class GPUStatus:
    available: bool
    device_type: str | None = None  # "cuda" or "mps"
    device_name: str | None = None
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None
    vram_free_mb: float | None = None
    cuda_version: str | None = None


def detect_gpu() -> GPUStatus:
    """Detect available GPU and report VRAM status."""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total = props.total_mem / (1024**2)
            used = torch.cuda.memory_allocated(device) / (1024**2)
            free = total - used
            cuda_ver = torch.version.cuda
            return GPUStatus(
                available=True,
                device_type="cuda",
                device_name=props.name,
                vram_total_mb=round(total, 1),
                vram_used_mb=round(used, 1),
                vram_free_mb=round(free, 1),
                cuda_version=cuda_ver,
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUStatus(
                available=True,
                device_type="mps",
                device_name="Apple Silicon (MPS)",
            )
    except ImportError:
        pass

    return GPUStatus(available=False)


def estimate_vram_usage(
    model_size_param: str,
    quantization_bits: int = 4,
    lora_rank: int = 16,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> dict[str, float]:
    """Estimate VRAM usage for training configuration.

    Returns dict with VRAM estimates in GB.
    """
    size_key = model_size_param.lower().replace("b", "") + "b"
    base_size = MODEL_SIZES_GB.get(size_key, 14.0)

    # Quantized model size
    if quantization_bits == 4:
        model_vram = base_size * (4 / 16)  # 4-bit = 1/4 of fp16
    elif quantization_bits == 8:
        model_vram = base_size * (8 / 16)  # 8-bit = 1/2 of fp16
    else:
        model_vram = base_size

    # LoRA adapter overhead (small)
    lora_overhead = (lora_rank * 2 * 0.001) * base_size  # rough estimate

    # Optimizer states (AdamW: 2x adapter params)
    optimizer_vram = lora_overhead * 3

    # Activation memory (rough: batch_size * seq_length factor)
    activation_vram = batch_size * seq_length * 0.0001 * (base_size / 14.0)

    # Gradient checkpointing saves ~60% activation memory
    activation_vram *= 0.4

    total = model_vram + lora_overhead + optimizer_vram + activation_vram

    return {
        "model_gb": round(model_vram, 2),
        "lora_adapter_gb": round(lora_overhead, 2),
        "optimizer_gb": round(optimizer_vram, 2),
        "activations_gb": round(activation_vram, 2),
        "total_estimated_gb": round(total, 2),
    }


def recommend_config(vram_mb: float | None) -> dict:
    """Recommend training configuration based on available VRAM."""
    if vram_mb is None:
        return {
            "recommendation": "No GPU detected. Use CPU for dataset prep and evaluation only.",
            "max_model": None,
            "quantization": None,
            "batch_size": None,
        }

    vram_gb = vram_mb / 1024

    if vram_gb >= 80:
        return {
            "recommendation": "High-end GPU. Can train 70B models with QLoRA.",
            "max_model": "70B",
            "quantization": 4,
            "batch_size": 8,
            "lora_rank": 64,
        }
    elif vram_gb >= 24:
        return {
            "recommendation": "Can train 7-13B models with QLoRA comfortably.",
            "max_model": "13B",
            "quantization": 4,
            "batch_size": 4,
            "lora_rank": 32,
        }
    elif vram_gb >= 16:
        return {
            "recommendation": "Can train 7B models with QLoRA.",
            "max_model": "7B",
            "quantization": 4,
            "batch_size": 4,
            "lora_rank": 16,
        }
    elif vram_gb >= 8:
        return {
            "recommendation": "Can train 3B models or 7B with aggressive quantization.",
            "max_model": "7B",
            "quantization": 4,
            "batch_size": 2,
            "lora_rank": 8,
        }
    else:
        return {
            "recommendation": "Limited VRAM. Try 1-3B models with 4-bit quantization.",
            "max_model": "3B",
            "quantization": 4,
            "batch_size": 1,
            "lora_rank": 8,
        }


def get_device(preference: str = "auto") -> str:
    """Get the best available device."""
    if preference != "auto":
        return preference

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
