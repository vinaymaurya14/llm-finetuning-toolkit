"""Adapter tracking and management via JSON-file registry."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models.schemas import ModelInfo

logger = logging.getLogger(__name__)


def _registry_path() -> Path:
    path = Path(settings.registry_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_registry() -> list[dict[str, Any]]:
    """Load the registry from disk."""
    path = _registry_path()
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_registry(entries: list[dict[str, Any]]):
    """Save the registry to disk."""
    path = _registry_path()
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def register_adapter(
    adapter_name: str,
    base_model: str,
    dataset: str,
    training_type: str,
    metrics: dict[str, Any],
    lora_config: dict[str, Any],
    adapter_path: str,
) -> ModelInfo:
    """Register a trained adapter in the registry."""
    # Calculate file size
    file_size_mb = None
    adapter_dir = Path(adapter_path)
    if adapter_dir.exists():
        total_bytes = sum(
            f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()
        )
        file_size_mb = round(total_bytes / (1024 * 1024), 2)

    info = ModelInfo(
        adapter_name=adapter_name,
        base_model=base_model,
        dataset=dataset,
        training_type=training_type,
        metrics=metrics,
        lora_config=lora_config,
        created_at=datetime.now().isoformat(),
        file_size_mb=file_size_mb,
        adapter_path=adapter_path,
    )

    entries = _load_registry()
    entries.append(info.model_dump())
    _save_registry(entries)

    logger.info(f"Registered adapter: {adapter_name}")
    return info


def list_adapters() -> list[ModelInfo]:
    """List all registered adapters."""
    entries = _load_registry()
    return [ModelInfo(**entry) for entry in entries]


def get_adapter(adapter_name: str) -> Optional[ModelInfo]:
    """Get a specific adapter by name."""
    entries = _load_registry()
    for entry in entries:
        if entry["adapter_name"] == adapter_name:
            return ModelInfo(**entry)
    return None


def delete_adapter(adapter_name: str) -> bool:
    """Remove an adapter from the registry (does not delete files)."""
    entries = _load_registry()
    original_len = len(entries)
    entries = [e for e in entries if e["adapter_name"] != adapter_name]

    if len(entries) < original_len:
        _save_registry(entries)
        logger.info(f"Deleted adapter from registry: {adapter_name}")
        return True

    return False


def get_adapter_by_path(adapter_path: str) -> Optional[ModelInfo]:
    """Find an adapter by its file path."""
    entries = _load_registry()
    for entry in entries:
        if entry["adapter_path"] == adapter_path:
            return ModelInfo(**entry)
    return None
