"""Evaluation metrics: perplexity, BLEU, ROUGE, accuracy."""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def compute_perplexity(losses: list[float]) -> float:
    """Compute perplexity from a list of cross-entropy losses.

    Perplexity = exp(mean(losses))
    """
    if not losses:
        return float("inf")
    avg_loss = sum(losses) / len(losses)
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float("inf")


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        logger.warning("NLTK not installed, returning 0.0 for BLEU")
        return 0.0

    if not predictions or not references:
        return 0.0

    # Tokenize
    refs = [[ref.split()] for ref in references]
    preds = [pred.split() for pred in predictions]

    smoothie = SmoothingFunction().method1
    try:
        score = corpus_bleu(refs, preds, smoothing_function=smoothie)
        return round(score, 4)
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed, returning zeros")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {
        key: round(sum(vals) / len(vals), 4) if vals else 0.0
        for key, vals in scores.items()
    }


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute exact-match accuracy."""
    if not predictions or not references:
        return 0.0

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return round(correct / len(predictions), 4)


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    losses: Optional[list[float]] = None,
    metrics: Optional[list[str]] = None,
) -> dict[str, float | dict]:
    """Compute all requested metrics.

    Args:
        predictions: Model-generated texts
        references: Ground-truth texts
        losses: Per-sample cross-entropy losses (for perplexity)
        metrics: List of metrics to compute (default: all)

    Returns:
        Dict of metric name -> score
    """
    if metrics is None:
        metrics = ["perplexity", "bleu", "rouge", "accuracy"]

    results = {}

    if "perplexity" in metrics and losses:
        results["perplexity"] = compute_perplexity(losses)

    if "bleu" in metrics:
        results["bleu"] = compute_bleu(predictions, references)

    if "rouge" in metrics:
        results["rouge"] = compute_rouge(predictions, references)

    if "accuracy" in metrics:
        results["accuracy"] = compute_accuracy(predictions, references)

    return results


def compute_improvement(
    base_scores: dict[str, float],
    finetuned_scores: dict[str, float],
) -> dict[str, float]:
    """Compute percentage improvement from base to fine-tuned."""
    improvement = {}
    for key in finetuned_scores:
        if key in base_scores and base_scores[key] != 0:
            if key == "perplexity":
                # Lower is better for perplexity
                improvement[key] = round(
                    ((base_scores[key] - finetuned_scores[key]) / base_scores[key]) * 100, 2
                )
            else:
                # Higher is better for other metrics
                improvement[key] = round(
                    ((finetuned_scores[key] - base_scores[key]) / base_scores[key]) * 100, 2
                )
    return improvement
