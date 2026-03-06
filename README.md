# LLM Fine-Tuning Toolkit

A full-lifecycle LLM fine-tuning pipeline: dataset preparation, QLoRA/LoRA training (SFT + DPO), evaluation, model registry, and inference — all through a FastAPI interface.

```
┌──────────────────────────────────────────────────────────────┐
│                   LLM Fine-Tuning Toolkit                    │
├──────────┬───────────┬────────────┬───────────┬─────────────┤
│ Dataset  │ Training  │ Evaluation │  Model    │  Inference  │
│ Engine   │ Pipeline  │ Framework  │ Registry  │  Server     │
│          │           │            │           │             │
│ • Load   │ • SFT     │ • PPL      │ • Track   │ • Load      │
│ • Format │ • DPO     │ • BLEU     │ • List    │ • Generate  │
│ • Filter │ • QLoRA   │ • ROUGE    │ • Export  │ • Stream    │
│ • Split  │ • Unsloth │ • Compare  │ • Delete  │ • Unload    │
└──────────┴───────────┴────────────┴───────────┴─────────────┘
         HuggingFace PEFT  •  TRL  •  bitsandbytes
```

## Features

- **Dataset Engine** — Load from HuggingFace Hub or local files (JSON/JSONL/CSV). Auto-detect and convert between Alpaca, ShareGPT, and Conversational formats. Quality filtering with deduplication, length checks, and toxicity detection.
- **Training Pipeline** — QLoRA 4-bit fine-tuning via PEFT + TRL. Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO). Unsloth integration for 2-5x speedup. Background job execution with real-time progress tracking.
- **Evaluation Framework** — Perplexity, BLEU, ROUGE, and exact-match accuracy. Side-by-side comparison of base vs fine-tuned models with improvement percentages.
- **Model Registry** — JSON-file-based adapter tracking with full lineage (base model, dataset, config, metrics). No external database needed.
- **Inference Server** — Load base model + LoRA adapter for interactive testing. Configurable generation parameters.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --port 8003
```

### Interactive API Docs

Open [http://localhost:8003/docs](http://localhost:8003/docs) for the Swagger UI.

## GPU Requirements

| Model Size | Quantization | Min VRAM | Recommended GPU |
|-----------|-------------|----------|-----------------|
| 3B | 4-bit QLoRA | 6 GB | RTX 3060 |
| 7B | 4-bit QLoRA | 10 GB | RTX 3080/4070 |
| 7B | 8-bit LoRA | 16 GB | RTX 4080/A4000 |
| 13B | 4-bit QLoRA | 16 GB | RTX 4090/A5000 |
| 70B | 4-bit QLoRA | 48 GB | A100/H100 |

**No GPU?** Dataset preparation, format conversion, quality filtering, and metric computation all work on CPU. Training endpoints will report GPU requirements if unavailable.

## API Reference

### Dataset Preparation

```bash
# Prepare a dataset from HuggingFace Hub
curl -X POST http://localhost:8003/datasets/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "source": "tatsu-lab/alpaca",
    "source_format": "auto",
    "target_format": "alpaca",
    "split_ratio": 0.9,
    "name": "alpaca_clean",
    "max_samples": 5000,
    "filters": {
      "min_length": 20,
      "max_length": 4096,
      "remove_duplicates": true
    }
  }'

# List all prepared datasets
curl http://localhost:8003/datasets
```

### Training

```bash
# Launch SFT training with QLoRA
curl -X POST http://localhost:8003/training/sft \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "alpaca_clean",
    "base_model": "mistralai/Mistral-7B-v0.3",
    "lora_config": {
      "preset": "balanced",
      "quantization_bits": 4
    },
    "training_args": {
      "num_epochs": 3,
      "learning_rate": 2e-4,
      "per_device_batch_size": 4
    }
  }'

# Check training status
curl http://localhost:8003/training/status/{job_id}

# Launch DPO alignment (after SFT)
curl -X POST http://localhost:8003/training/dpo \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "preference_data",
    "sft_adapter_path": "./outputs/sft_job_xxx/final_adapter",
    "beta": 0.1
  }'
```

### Evaluation

```bash
# Evaluate a fine-tuned adapter
curl -X POST http://localhost:8003/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_path": "./outputs/sft_job_xxx/final_adapter",
    "dataset_name": "alpaca_clean",
    "metrics": ["perplexity", "bleu", "rouge"],
    "num_samples": 100
  }'

# Compare base vs fine-tuned
curl -X POST http://localhost:8003/evaluate/compare \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_path": "./outputs/sft_job_xxx/final_adapter",
    "dataset_name": "alpaca_clean",
    "num_samples": 50
  }'
```

### Inference

```bash
# Generate text with fine-tuned model
curl -X POST http://localhost:8003/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
    "adapter_path": "./outputs/sft_job_xxx/final_adapter",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Model Registry

```bash
# List all registered adapters
curl http://localhost:8003/models

# Get adapter details
curl http://localhost:8003/models/{adapter_name}

# Delete adapter from registry
curl -X DELETE http://localhost:8003/models/{adapter_name}
```

## Example Training Workflow

```
1. Prepare Dataset
   POST /datasets/prepare
   └── Load tatsu-lab/alpaca → filter → split (90/10)

2. Fine-Tune with QLoRA
   POST /training/sft
   └── Mistral-7B + 4-bit QLoRA (rank=16) → 3 epochs

3. Evaluate
   POST /evaluate
   └── Perplexity: 4.2, BLEU: 0.42, ROUGE-L: 0.58

4. Compare
   POST /evaluate/compare
   └── Base PPL: 8.1 → Fine-tuned PPL: 4.2 (48% improvement)

5. Inference
   POST /inference/generate
   └── Interactive testing with the fine-tuned model
```

## LoRA Presets

| Preset | Rank | Alpha | Dropout | Use Case |
|--------|------|-------|---------|----------|
| `efficient` | 8 | 16 | 0.05 | Quick experiments, low VRAM |
| `balanced` | 16 | 32 | 0.05 | Recommended default |
| `quality` | 64 | 128 | 0.10 | Best quality, more VRAM |

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

All settings can be overridden via environment variables with `LFT_` prefix.

## Docker

```bash
# Build and run with GPU support
docker compose up -d

# View logs
docker compose logs -f toolkit
```

## Testing

```bash
# Run all tests (CPU-only, no GPU needed)
pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Fine-tuning | HuggingFace Transformers + PEFT |
| Training | TRL (SFTTrainer, DPOTrainer) |
| Acceleration | Unsloth (optional, 2-5x speedup) |
| Quantization | bitsandbytes (4-bit/8-bit QLoRA) |
| Evaluation | NLTK (BLEU), rouge-score, PyTorch |
| Data | HuggingFace Datasets, pandas |
| Validation | Pydantic v2 |
| Testing | pytest |

## Project Structure

```
llm-finetuning-toolkit/
├── app/
│   ├── main.py                    # FastAPI app + all routes
│   ├── config.py                  # Pydantic settings
│   ├── models/schemas.py          # Request/response models
│   ├── services/
│   │   ├── dataset_engine.py      # Dataset loading & preparation
│   │   ├── trainer.py             # SFT + DPO training
│   │   ├── evaluator.py           # Evaluation pipeline
│   │   ├── model_registry.py      # Adapter tracking
│   │   └── inference.py           # Model loading & generation
│   ├── core/
│   │   ├── dataset_formats.py     # Format converters
│   │   ├── quality_filters.py     # Quality filtering
│   │   ├── lora_config.py         # LoRA config builder
│   │   ├── training_callbacks.py  # Progress tracking
│   │   └── eval_metrics.py        # Metric computation
│   └── utils/
│       ├── gpu_utils.py           # GPU detection & VRAM estimation
│       └── text_utils.py          # Text processing helpers
├── tests/
│   ├── test_dataset_engine.py     # Format + filter tests
│   ├── test_evaluator.py          # Metric calculation tests
│   └── test_api.py                # API endpoint tests
├── notebooks/
│   └── quickstart.ipynb           # Interactive demo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## License

MIT
