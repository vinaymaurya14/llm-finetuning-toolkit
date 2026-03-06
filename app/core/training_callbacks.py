"""Custom training callbacks for logging and progress tracking."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Track training progress and metrics."""
    job_id: str
    total_steps: int = 0
    current_step: int = 0
    current_epoch: float = 0.0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    best_eval_loss: Optional[float] = None
    start_time: Optional[float] = None
    metrics_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.current_step == 0 or self.start_time is None:
            return None
        rate = self.elapsed_seconds / self.current_step
        remaining = self.total_steps - self.current_step
        return rate * remaining

    def update(self, logs: dict[str, Any]):
        """Update progress from training logs."""
        if "loss" in logs:
            self.train_loss = logs["loss"]
        if "eval_loss" in logs:
            self.eval_loss = logs["eval_loss"]
            if self.best_eval_loss is None or logs["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = logs["eval_loss"]
        if "learning_rate" in logs:
            self.learning_rate = logs["learning_rate"]
        if "epoch" in logs:
            self.current_epoch = logs["epoch"]

        self.metrics_history.append({
            "step": self.current_step,
            "epoch": self.current_epoch,
            **logs,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "progress_pct": round(self.progress_pct, 1),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": round(self.current_epoch, 2),
            "train_loss": round(self.train_loss, 4) if self.train_loss else None,
            "eval_loss": round(self.eval_loss, 4) if self.eval_loss else None,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
        }


# Store active training progress globally
_active_jobs: dict[str, TrainingProgress] = {}


def create_progress_tracker(job_id: str, total_steps: int) -> TrainingProgress:
    """Create and register a new progress tracker."""
    progress = TrainingProgress(
        job_id=job_id,
        total_steps=total_steps,
        start_time=time.time(),
    )
    _active_jobs[job_id] = progress
    return progress


def get_progress(job_id: str) -> Optional[TrainingProgress]:
    """Get progress tracker for a job."""
    return _active_jobs.get(job_id)


def remove_progress(job_id: str):
    """Remove a completed job's progress tracker."""
    _active_jobs.pop(job_id, None)


def build_trainer_callbacks(progress: TrainingProgress) -> list:
    """Build HuggingFace TrainerCallback instances for progress tracking.

    Returns a list of callback class instances that work with HF Trainer.
    """
    try:
        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def __init__(self, progress_tracker: TrainingProgress):
                self.progress = progress_tracker

            def on_step_end(self, args, state, control, **kwargs):
                self.progress.current_step = state.global_step
                if state.log_history:
                    self.progress.update(state.log_history[-1])

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    self.progress.update(logs)
                    logger.info(
                        f"[{self.progress.job_id}] Step {state.global_step}/{self.progress.total_steps} "
                        f"| Loss: {self.progress.train_loss:.4f} "
                        f"| LR: {self.progress.learning_rate:.2e}"
                    )

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    self.progress.update(metrics)
                    logger.info(
                        f"[{self.progress.job_id}] Eval loss: {metrics.get('eval_loss', 'N/A')}"
                    )

        return [ProgressCallback(progress)]

    except ImportError:
        logger.warning("transformers not installed, callbacks unavailable")
        return []
