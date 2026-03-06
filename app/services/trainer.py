"""SFT + DPO training orchestration service."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.core.lora_config import build_lora_config, build_quantization_config, validate_config_for_gpu
from app.core.training_callbacks import create_progress_tracker, get_progress, remove_progress
from app.models.schemas import (
    DPOTrainingRequest,
    SFTTrainingRequest,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from app.services.model_registry import register_adapter
from app.utils.gpu_utils import detect_gpu

logger = logging.getLogger(__name__)

# In-memory job store
_jobs: dict[str, TrainingJob] = {}


def _generate_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def get_job(job_id: str) -> Optional[TrainingJob]:
    """Get training job by ID."""
    job = _jobs.get(job_id)

    # Sync with progress tracker
    if job and job.status == TrainingStatus.RUNNING:
        progress = get_progress(job_id)
        if progress:
            job.progress = progress.progress_pct
            job.metrics = TrainingMetrics(
                train_loss=progress.train_loss,
                eval_loss=progress.eval_loss,
                learning_rate=progress.learning_rate,
                epoch=progress.current_epoch,
                step=progress.current_step,
                total_steps=progress.total_steps,
            )

    return job


def list_jobs() -> list[TrainingJob]:
    """List all training jobs."""
    # Sync progress for running jobs
    for job_id in _jobs:
        get_job(job_id)
    return list(_jobs.values())


async def start_sft_training(request: SFTTrainingRequest) -> TrainingJob:
    """Launch SFT (Supervised Fine-Tuning) training job.

    Pipeline:
    1. Load base model with QLoRA quantization
    2. Apply LoRA adapters
    3. Load and format dataset
    4. Train with SFTTrainer from TRL
    5. Save adapter weights
    """
    job_id = _generate_job_id()
    base_model = request.base_model or settings.base_model

    # Build configs
    lora_cfg = build_lora_config(
        preset=request.lora_config.preset.value,
        rank=request.lora_config.rank,
        alpha=request.lora_config.alpha,
        dropout=request.lora_config.dropout,
        target_modules=request.lora_config.target_modules,
    )

    quant_bits = request.lora_config.quantization_bits

    # Validate GPU availability
    validation = validate_config_for_gpu(
        model_name=base_model,
        lora_config=lora_cfg,
        quantization_bits=quant_bits,
        batch_size=request.training_args.per_device_batch_size or settings.per_device_batch_size,
        seq_length=request.training_args.max_seq_length or settings.max_seq_length,
    )

    job = TrainingJob(
        job_id=job_id,
        job_type="sft",
        status=TrainingStatus.PENDING,
        config={
            "base_model": base_model,
            "dataset": request.dataset_name,
            "lora": lora_cfg,
            "quantization_bits": quant_bits,
            "training_args": request.training_args.model_dump(exclude_none=True),
            "use_unsloth": request.use_unsloth,
        },
        started_at=datetime.now().isoformat(),
    )

    if not validation["valid"]:
        job.status = TrainingStatus.FAILED
        job.error_message = "; ".join(validation["warnings"])
        _jobs[job_id] = job
        return job

    _jobs[job_id] = job

    # Run training in background
    import asyncio
    asyncio.create_task(_run_sft_training(job, request, lora_cfg, quant_bits))

    return job


async def _run_sft_training(
    job: TrainingJob,
    request: SFTTrainingRequest,
    lora_cfg: dict[str, Any],
    quant_bits: int,
):
    """Execute the SFT training pipeline."""
    import asyncio

    try:
        job.status = TrainingStatus.RUNNING
        base_model = request.base_model or settings.base_model

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        adapter_path = await loop.run_in_executor(
            None,
            _execute_sft,
            job,
            base_model,
            request,
            lora_cfg,
            quant_bits,
        )

        job.status = TrainingStatus.COMPLETED
        job.adapter_path = adapter_path
        job.progress = 100.0
        job.completed_at = datetime.now().isoformat()

        # Register adapter
        progress = get_progress(job.job_id)
        metrics = {}
        if progress:
            metrics = {
                "final_train_loss": progress.train_loss,
                "final_eval_loss": progress.eval_loss,
                "total_steps": progress.total_steps,
            }

        register_adapter(
            adapter_name=Path(adapter_path).name,
            base_model=base_model,
            dataset=request.dataset_name,
            training_type="sft",
            metrics=metrics,
            lora_config=lora_cfg,
            adapter_path=adapter_path,
        )

        logger.info(f"SFT training completed: {job.job_id}")

    except Exception as e:
        job.status = TrainingStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now().isoformat()
        logger.error(f"SFT training failed: {e}")
    finally:
        remove_progress(job.job_id)


def _execute_sft(
    job: TrainingJob,
    base_model: str,
    request: SFTTrainingRequest,
    lora_cfg: dict[str, Any],
    quant_bits: int,
) -> str:
    """Synchronous SFT execution (runs in thread pool)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from app.services.dataset_engine import load_dataset_split
    from app.core.training_callbacks import create_progress_tracker, build_trainer_callbacks

    # Output path
    output_dir = settings.output_path / f"sft_{job.job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model_kwargs = {"trust_remote_code": True}

    if quant_bits in (4, 8):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(**build_quantization_config(quant_bits))
        model_kwargs["quantization_config"] = bnb_config

    # Try Unsloth first for faster training
    model = None
    if request.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=request.training_args.max_seq_length or settings.max_seq_length,
                load_in_4bit=(quant_bits == 4),
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                lora_dropout=lora_cfg["lora_dropout"],
                target_modules=lora_cfg["target_modules"],
            )
            logger.info("Using Unsloth for accelerated training")
        except ImportError:
            logger.info("Unsloth not available, falling back to standard PEFT")
            model = None

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if quant_bits in (4, 8):
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Load dataset
    train_data = load_dataset_split(request.dataset_name, "train")
    val_data = load_dataset_split(request.dataset_name, "val")

    # Convert to HF Dataset
    from datasets import Dataset
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Format dataset for SFT
    max_seq = request.training_args.max_seq_length or settings.max_seq_length
    lr = request.training_args.learning_rate or settings.learning_rate
    epochs = request.training_args.num_epochs or settings.num_epochs
    batch_size = request.training_args.per_device_batch_size or settings.per_device_batch_size
    grad_accum = request.training_args.gradient_accumulation_steps or settings.gradient_accumulation_steps

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=request.training_args.warmup_ratio or settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.max_grad_norm,
        lr_scheduler_type=settings.lr_scheduler_type,
        logging_steps=request.training_args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=request.training_args.fp16,
        gradient_checkpointing=request.training_args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    # Estimate total steps
    total_steps = (len(train_data) // (batch_size * grad_accum)) * epochs
    progress = create_progress_tracker(job.job_id, total_steps)
    callbacks = build_trainer_callbacks(progress)

    # SFT Trainer
    from trl import SFTTrainer

    # Determine text field based on format
    def formatting_func(examples):
        texts = []
        for i in range(len(examples.get("instruction", examples.get("messages", [""])))):
            if "instruction" in examples:
                instruction = examples["instruction"][i]
                inp = examples.get("input", [""] * len(examples["instruction"]))[i]
                output = examples["output"][i]
                if inp:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            elif "messages" in examples:
                msgs = examples["messages"][i]
                parts = []
                for msg in msgs:
                    parts.append(f"<|{msg['role']}|>\n{msg['content']}")
                text = "\n".join(parts)
            elif "conversations" in examples:
                convos = examples["conversations"][i]
                parts = []
                for turn in convos:
                    role = "user" if turn["from"] == "human" else "assistant"
                    parts.append(f"<|{role}|>\n{turn['value']}")
                text = "\n".join(parts)
            else:
                text = str(examples.get("text", [""])[i])
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=max_seq,
        callbacks=callbacks,
    )

    # Train
    if request.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=request.resume_from_checkpoint)
    else:
        trainer.train()

    # Save adapter
    adapter_path = str(output_dir / "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    return adapter_path


async def start_dpo_training(request: DPOTrainingRequest) -> TrainingJob:
    """Launch DPO (Direct Preference Optimization) training job."""
    job_id = _generate_job_id()
    base_model = request.base_model or settings.base_model

    lora_cfg = build_lora_config(
        preset=request.lora_config.preset.value,
        rank=request.lora_config.rank,
        alpha=request.lora_config.alpha,
        dropout=request.lora_config.dropout,
        target_modules=request.lora_config.target_modules,
    )

    quant_bits = request.lora_config.quantization_bits

    validation = validate_config_for_gpu(
        model_name=base_model,
        lora_config=lora_cfg,
        quantization_bits=quant_bits,
        batch_size=request.training_args.per_device_batch_size or settings.per_device_batch_size,
    )

    job = TrainingJob(
        job_id=job_id,
        job_type="dpo",
        status=TrainingStatus.PENDING,
        config={
            "base_model": base_model,
            "dataset": request.dataset_name,
            "lora": lora_cfg,
            "quantization_bits": quant_bits,
            "beta": request.beta,
            "sft_adapter": request.sft_adapter_path,
            "training_args": request.training_args.model_dump(exclude_none=True),
        },
        started_at=datetime.now().isoformat(),
    )

    if not validation["valid"]:
        job.status = TrainingStatus.FAILED
        job.error_message = "; ".join(validation["warnings"])
        _jobs[job_id] = job
        return job

    _jobs[job_id] = job

    import asyncio
    asyncio.create_task(_run_dpo_training(job, request, lora_cfg, quant_bits))

    return job


async def _run_dpo_training(
    job: TrainingJob,
    request: DPOTrainingRequest,
    lora_cfg: dict[str, Any],
    quant_bits: int,
):
    """Execute the DPO training pipeline."""
    import asyncio

    try:
        job.status = TrainingStatus.RUNNING
        base_model = request.base_model or settings.base_model

        loop = asyncio.get_event_loop()
        adapter_path = await loop.run_in_executor(
            None,
            _execute_dpo,
            job,
            base_model,
            request,
            lora_cfg,
            quant_bits,
        )

        job.status = TrainingStatus.COMPLETED
        job.adapter_path = adapter_path
        job.progress = 100.0
        job.completed_at = datetime.now().isoformat()

        progress = get_progress(job.job_id)
        metrics = {}
        if progress:
            metrics = {
                "final_train_loss": progress.train_loss,
                "final_eval_loss": progress.eval_loss,
            }

        register_adapter(
            adapter_name=Path(adapter_path).name,
            base_model=base_model,
            dataset=request.dataset_name,
            training_type="dpo",
            metrics=metrics,
            lora_config=lora_cfg,
            adapter_path=adapter_path,
        )

        logger.info(f"DPO training completed: {job.job_id}")

    except Exception as e:
        job.status = TrainingStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now().isoformat()
        logger.error(f"DPO training failed: {e}")
    finally:
        remove_progress(job.job_id)


def _execute_dpo(
    job: TrainingJob,
    base_model: str,
    request: DPOTrainingRequest,
    lora_cfg: dict[str, Any],
    quant_bits: int,
) -> str:
    """Synchronous DPO execution (runs in thread pool)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer

    from app.services.dataset_engine import load_dataset_split
    from app.core.training_callbacks import create_progress_tracker, build_trainer_callbacks

    output_dir = settings.output_path / f"dpo_{job.job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {"trust_remote_code": True}
    if quant_bits in (4, 8):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(**build_quantization_config(quant_bits))
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if quant_bits in (4, 8):
        model = prepare_model_for_kbit_training(model)

    # Load SFT adapter if provided
    if request.sft_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, request.sft_adapter_path)
        model = model.merge_and_unload()

    # Apply LoRA for DPO
    peft_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, peft_config)

    # Load DPO dataset
    train_data = load_dataset_split(request.dataset_name, "train")
    val_data = load_dataset_split(request.dataset_name, "val")

    from datasets import Dataset
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    lr = request.training_args.learning_rate or settings.learning_rate
    epochs = request.training_args.num_epochs or settings.num_epochs
    batch_size = request.training_args.per_device_batch_size or settings.per_device_batch_size
    grad_accum = request.training_args.gradient_accumulation_steps or settings.gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=request.training_args.warmup_ratio or settings.warmup_ratio,
        logging_steps=request.training_args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=request.training_args.fp16,
        gradient_checkpointing=request.training_args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    total_steps = (len(train_data) // (batch_size * grad_accum)) * epochs
    progress = create_progress_tracker(job.job_id, total_steps)
    callbacks = build_trainer_callbacks(progress)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        beta=request.beta,
        callbacks=callbacks,
    )

    trainer.train()

    adapter_path = str(output_dir / "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    return adapter_path
