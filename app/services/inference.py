"""Model loading and text generation service."""

import logging
import time
from typing import Any, Optional

from app.config import settings
from app.models.schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

# Cache loaded models to avoid reloading
_loaded_models: dict[str, Any] = {}
_loaded_tokenizers: dict[str, Any] = {}


def _model_key(base_model: str, adapter_path: Optional[str] = None) -> str:
    """Create a cache key for a model configuration."""
    return f"{base_model}::{adapter_path or 'base'}"


def load_model(
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """Load a model with optional LoRA adapter. Uses cache if available."""
    base_model = base_model or settings.base_model
    key = _model_key(base_model, adapter_path)

    if key in _loaded_models:
        logger.info(f"Using cached model: {key}")
        return _loaded_models[key], _loaded_tokenizers[key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    _loaded_models[key] = model
    _loaded_tokenizers[key] = tokenizer

    return model, tokenizer


def unload_model(
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """Unload a model from cache to free GPU memory."""
    base_model = base_model or settings.base_model
    key = _model_key(base_model, adapter_path)

    if key in _loaded_models:
        del _loaded_models[key]
        del _loaded_tokenizers[key]

        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(f"Unloaded model: {key}")
        return True

    return False


def unload_all():
    """Unload all cached models."""
    _loaded_models.clear()
    _loaded_tokenizers.clear()

    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("Unloaded all models")


async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Generate text using a loaded model.

    Supports both base model and base + adapter inference.
    """
    import asyncio

    base_model = request.base_model or settings.base_model

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _execute_generation,
            request,
            base_model,
        )
        return result
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependencies for inference: {e}. "
            "Install with: pip install torch transformers"
        )


def _execute_generation(request: GenerateRequest, base_model: str) -> GenerateResponse:
    """Synchronous text generation."""
    import torch

    start_time = time.time()

    model, tokenizer = load_model(base_model, request.adapter_path)

    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_seq_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    elapsed_ms = (time.time() - start_time) * 1000

    return GenerateResponse(
        text=text.strip(),
        tokens_used=len(generated_tokens),
        generation_time_ms=round(elapsed_ms, 1),
        model=base_model,
        adapter=request.adapter_path,
    )


def get_loaded_models() -> list[str]:
    """Get list of currently loaded model keys."""
    return list(_loaded_models.keys())
