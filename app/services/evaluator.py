"""Evaluation pipeline for fine-tuned models."""

import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.core.eval_metrics import compute_all_metrics, compute_improvement
from app.models.schemas import CompareResult, EvaluateRequest, EvaluationResult

logger = logging.getLogger(__name__)


def _load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str] = None,
):
    """Load a model with optional LoRA adapter for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded adapter from {adapter_path}")

    model.eval()
    return model, tokenizer


def _generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate predictions for a list of prompts."""
    import torch

    predictions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(text.strip())

    return predictions


def _compute_losses(
    model,
    tokenizer,
    texts: list[str],
) -> list[float]:
    """Compute per-sample cross-entropy losses."""
    import torch

    losses = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    return losses


async def evaluate_model(request: EvaluateRequest) -> EvaluationResult:
    """Run evaluation suite on a trained adapter.

    Computes requested metrics against a validation dataset.
    """
    import asyncio

    base_model = request.base_model or settings.base_model

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _execute_evaluation,
            request,
            base_model,
        )
        return result
    except ImportError as e:
        logger.warning(f"Missing dependencies for evaluation: {e}")
        return EvaluationResult(
            adapter_path=request.adapter_path,
            samples=[{"error": f"Missing dependencies: {e}"}],
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return EvaluationResult(
            adapter_path=request.adapter_path,
            samples=[{"error": str(e)}],
        )


def _execute_evaluation(request: EvaluateRequest, base_model: str) -> EvaluationResult:
    """Synchronous evaluation execution."""
    from app.services.dataset_engine import load_dataset_split

    model, tokenizer = _load_model_and_tokenizer(base_model, request.adapter_path)

    # Load evaluation data
    prompts = []
    references = []

    if request.dataset_name:
        val_data = load_dataset_split(request.dataset_name, "val")
        val_data = val_data[: request.num_samples]

        for row in val_data:
            if "instruction" in row:
                inp = row.get("input", "")
                prompt = f"### Instruction:\n{row['instruction']}"
                if inp:
                    prompt += f"\n\n### Input:\n{inp}"
                prompt += "\n\n### Response:\n"
                prompts.append(prompt)
                references.append(row.get("output", ""))
            elif "messages" in row:
                user_msgs = [m for m in row["messages"] if m["role"] == "user"]
                asst_msgs = [m for m in row["messages"] if m["role"] == "assistant"]
                if user_msgs and asst_msgs:
                    prompts.append(user_msgs[-1]["content"])
                    references.append(asst_msgs[-1]["content"])

    # Generate predictions
    predictions = _generate_predictions(model, tokenizer, prompts)

    # Compute losses for perplexity
    losses = None
    if "perplexity" in request.metrics:
        full_texts = [f"{p}{r}" for p, r in zip(prompts, references)]
        losses = _compute_losses(model, tokenizer, full_texts[:request.num_samples])

    # Compute metrics
    metrics = compute_all_metrics(
        predictions=predictions,
        references=references,
        losses=losses,
        metrics=request.metrics,
    )

    # Sample outputs for qualitative review
    samples = []
    for i in range(min(5, len(prompts))):
        samples.append({
            "prompt": prompts[i],
            "reference": references[i],
            "generated": predictions[i],
        })

    return EvaluationResult(
        adapter_path=request.adapter_path,
        perplexity=metrics.get("perplexity"),
        bleu=metrics.get("bleu"),
        rouge=metrics.get("rouge"),
        accuracy=metrics.get("accuracy"),
        samples=samples,
    )


async def compare_models(
    adapter_path: str,
    base_model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    num_samples: int = 50,
) -> CompareResult:
    """Compare base model vs fine-tuned model on the same benchmark."""
    import asyncio

    base_model = base_model or settings.base_model

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _execute_comparison,
            adapter_path,
            base_model,
            dataset_name,
            num_samples,
        )
        return result
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return CompareResult(
            base_scores={},
            finetuned_scores={},
            improvement_pct={},
            sample_comparisons=[{"error": str(e)}],
        )


def _execute_comparison(
    adapter_path: str,
    base_model: str,
    dataset_name: Optional[str],
    num_samples: int,
) -> CompareResult:
    """Synchronous comparison execution."""
    from app.services.dataset_engine import load_dataset_split

    # Load evaluation data
    prompts = []
    references = []

    if dataset_name:
        val_data = load_dataset_split(dataset_name, "val")
        val_data = val_data[:num_samples]

        for row in val_data:
            if "instruction" in row:
                inp = row.get("input", "")
                prompt = f"### Instruction:\n{row['instruction']}"
                if inp:
                    prompt += f"\n\n### Input:\n{inp}"
                prompt += "\n\n### Response:\n"
                prompts.append(prompt)
                references.append(row.get("output", ""))

    # Evaluate base model
    logger.info("Evaluating base model...")
    base_model_obj, tokenizer = _load_model_and_tokenizer(base_model)
    base_predictions = _generate_predictions(base_model_obj, tokenizer, prompts)
    base_losses = _compute_losses(
        base_model_obj, tokenizer,
        [f"{p}{r}" for p, r in zip(prompts, references)],
    )
    base_metrics = compute_all_metrics(base_predictions, references, base_losses)

    # Free base model
    del base_model_obj
    _try_free_gpu_memory()

    # Evaluate fine-tuned model
    logger.info("Evaluating fine-tuned model...")
    ft_model, tokenizer = _load_model_and_tokenizer(base_model, adapter_path)
    ft_predictions = _generate_predictions(ft_model, tokenizer, prompts)
    ft_losses = _compute_losses(
        ft_model, tokenizer,
        [f"{p}{r}" for p, r in zip(prompts, references)],
    )
    ft_metrics = compute_all_metrics(ft_predictions, references, ft_losses)

    del ft_model
    _try_free_gpu_memory()

    # Flatten metrics for comparison
    base_flat = _flatten_metrics(base_metrics)
    ft_flat = _flatten_metrics(ft_metrics)

    improvement = compute_improvement(base_flat, ft_flat)

    # Sample comparisons
    comparisons = []
    for i in range(min(5, len(prompts))):
        comparisons.append({
            "prompt": prompts[i],
            "reference": references[i],
            "base_output": base_predictions[i],
            "finetuned_output": ft_predictions[i],
        })

    return CompareResult(
        base_scores=base_flat,
        finetuned_scores=ft_flat,
        improvement_pct=improvement,
        sample_comparisons=comparisons,
    )


def _flatten_metrics(metrics: dict) -> dict[str, float]:
    """Flatten nested metric dicts into a single-level dict."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, (int, float)):
            flat[key] = value
    return flat


def _try_free_gpu_memory():
    """Attempt to free GPU memory."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
