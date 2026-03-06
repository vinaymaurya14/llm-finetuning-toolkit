"""Dataset loading, formatting, validation, and preparation service."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.core.dataset_formats import convert_format, detect_format, get_text_for_filtering
from app.core.quality_filters import QualityFilter
from app.models.schemas import DatasetFormat, DatasetInfo, QualityFilterConfig

logger = logging.getLogger(__name__)

# In-memory dataset registry
_datasets: dict[str, DatasetInfo] = {}


def _load_from_huggingface(
    source: str,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset

        kwargs = {}
        if subset:
            kwargs["name"] = subset

        ds = load_dataset(source, split="train", **kwargs)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        return [dict(row) for row in ds]

    except Exception as e:
        raise ValueError(f"Failed to load HuggingFace dataset '{source}': {e}")


def _load_from_file(source: str, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    """Load dataset from local JSON, JSONL, or CSV file."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source}")

    data = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        data = raw if isinstance(raw, list) else [raw]
    elif suffix in (".jsonl", ".ndjson"):
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif suffix == ".csv":
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json, .jsonl, or .csv")

    if max_samples and max_samples < len(data):
        data = data[:max_samples]

    return data


def _is_hf_dataset(source: str) -> bool:
    """Check if the source looks like a HuggingFace dataset identifier."""
    return "/" in source and not os.path.exists(source)


def prepare_dataset(
    source: str,
    source_format: DatasetFormat = DatasetFormat.AUTO,
    target_format: DatasetFormat = DatasetFormat.ALPACA,
    split_ratio: float = 0.9,
    filters: Optional[QualityFilterConfig] = None,
    name: Optional[str] = None,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> DatasetInfo:
    """Full dataset preparation pipeline.

    1. Load data from source
    2. Detect/convert format
    3. Apply quality filters
    4. Split into train/val
    5. Save to disk
    6. Register metadata
    """
    # Step 1: Load data
    logger.info(f"Loading dataset from: {source}")
    if _is_hf_dataset(source):
        raw_data = _load_from_huggingface(source, subset=subset, max_samples=max_samples)
    else:
        raw_data = _load_from_file(source, max_samples=max_samples)

    if not raw_data:
        raise ValueError("Dataset is empty after loading")

    logger.info(f"Loaded {len(raw_data)} samples")

    # Step 2: Detect format
    src_fmt = source_format.value
    if src_fmt == "auto":
        src_fmt = detect_format(raw_data[0])
        logger.info(f"Auto-detected format: {src_fmt}")

    tgt_fmt = target_format.value

    # Step 3: Convert format
    converted = []
    conversion_errors = 0
    for row in raw_data:
        try:
            converted_row = convert_format(row, src_fmt, tgt_fmt)
            converted.append(converted_row)
        except Exception as e:
            conversion_errors += 1
            if conversion_errors <= 3:
                logger.warning(f"Format conversion error: {e}")

    if conversion_errors:
        logger.warning(f"Skipped {conversion_errors} rows due to conversion errors")

    # Step 4: Apply quality filters
    if filters:
        qf = QualityFilter(
            min_length=filters.min_length,
            max_length=filters.max_length,
            remove_duplicates=filters.remove_duplicates,
            near_dedup=filters.near_dedup,
            min_quality_score=filters.min_quality_score,
        )

        filtered = []
        for row in converted:
            text = get_text_for_filtering(row, tgt_fmt)
            if qf.passes(text):
                filtered.append(row)

        logger.info(f"After filtering: {len(filtered)}/{len(converted)} samples kept")
        logger.info(f"Filter stats: {qf.stats.to_dict()}")
        converted = filtered

    if not converted:
        raise ValueError("No samples remaining after filtering")

    # Step 5: Train/val split
    split_idx = int(len(converted) * split_ratio)
    train_data = converted[:split_idx]
    val_data = converted[split_idx:]

    # Step 6: Save to disk
    dataset_name = name or _generate_dataset_name(source)
    save_dir = settings.dataset_path / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_data, save_dir / "train.jsonl")
    _save_jsonl(val_data, save_dir / "val.jsonl")

    # Save metadata
    columns = list(converted[0].keys()) if converted else []
    info = DatasetInfo(
        name=dataset_name,
        num_samples=len(converted),
        format=DatasetFormat(tgt_fmt),
        columns=columns,
        split_sizes={"train": len(train_data), "val": len(val_data)},
        source=source,
        created_at=datetime.now().isoformat(),
        path=str(save_dir),
    )

    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(info.model_dump(), f, indent=2)

    _datasets[dataset_name] = info
    logger.info(f"Dataset '{dataset_name}' saved to {save_dir}")
    return info


def list_datasets() -> list[DatasetInfo]:
    """List all prepared datasets."""
    # Also scan disk for previously saved datasets
    dataset_dir = settings.dataset_path
    if dataset_dir.exists():
        for sub in dataset_dir.iterdir():
            if sub.is_dir() and sub.name not in _datasets:
                meta_file = sub / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    _datasets[sub.name] = DatasetInfo(**meta)

    return list(_datasets.values())


def get_dataset(name: str) -> Optional[DatasetInfo]:
    """Get a specific dataset by name."""
    if name not in _datasets:
        list_datasets()  # Refresh from disk
    return _datasets.get(name)


def load_dataset_split(name: str, split: str = "train") -> list[dict[str, Any]]:
    """Load a dataset split from disk."""
    info = get_dataset(name)
    if info is None:
        raise FileNotFoundError(f"Dataset '{name}' not found")

    split_path = Path(info.path) / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split '{split}' not found for dataset '{name}'")

    with open(split_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(data: list[dict], path: Path):
    """Save data as JSONL."""
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def _generate_dataset_name(source: str) -> str:
    """Generate a dataset name from the source."""
    name = source.replace("/", "_").replace("\\", "_")
    name = "".join(c for c in name if c.isalnum() or c in "_-.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"
