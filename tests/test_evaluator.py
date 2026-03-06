"""Tests for evaluation metrics with mock data (CPU-only)."""

import math

import pytest

from app.core.eval_metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_bleu,
    compute_improvement,
    compute_perplexity,
    compute_rouge,
)


class TestPerplexity:
    def test_basic_perplexity(self):
        losses = [1.0, 2.0, 3.0]
        ppl = compute_perplexity(losses)
        expected = math.exp(2.0)  # mean = 2.0
        assert abs(ppl - expected) < 0.01

    def test_zero_loss(self):
        ppl = compute_perplexity([0.0])
        assert ppl == 1.0  # exp(0) = 1

    def test_empty_losses(self):
        ppl = compute_perplexity([])
        assert ppl == float("inf")

    def test_single_loss(self):
        ppl = compute_perplexity([2.5])
        assert abs(ppl - math.exp(2.5)) < 0.01

    def test_high_loss_overflow(self):
        ppl = compute_perplexity([1000.0])
        assert ppl == float("inf")


class TestBLEU:
    def test_perfect_bleu(self):
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        score = compute_bleu(predictions, references)
        assert score > 0.9

    def test_zero_bleu(self):
        predictions = ["completely different text here"]
        references = ["nothing in common at all"]
        score = compute_bleu(predictions, references)
        assert score < 0.5

    def test_empty_inputs(self):
        assert compute_bleu([], []) == 0.0
        assert compute_bleu(["text"], []) == 0.0

    def test_partial_match(self):
        predictions = ["the cat sat on a mat"]
        references = ["the cat sat on the mat"]
        score = compute_bleu(predictions, references)
        assert 0.3 < score < 1.0


class TestROUGE:
    def test_rouge_scores(self):
        predictions = ["the cat is on the mat"]
        references = ["the cat is on the mat"]
        scores = compute_rouge(predictions, references)
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
        # Perfect match should give high scores
        assert scores["rouge1"] > 0.9

    def test_rouge_empty(self):
        scores = compute_rouge([], [])
        assert scores["rouge1"] == 0.0

    def test_rouge_partial(self):
        predictions = ["a cat on a mat"]
        references = ["the cat is on the mat"]
        scores = compute_rouge(predictions, references)
        assert 0.3 < scores["rouge1"] < 1.0


class TestAccuracy:
    def test_perfect_accuracy(self):
        preds = ["yes", "no", "maybe"]
        refs = ["yes", "no", "maybe"]
        assert compute_accuracy(preds, refs) == 1.0

    def test_zero_accuracy(self):
        preds = ["a", "b", "c"]
        refs = ["x", "y", "z"]
        assert compute_accuracy(preds, refs) == 0.0

    def test_partial_accuracy(self):
        preds = ["yes", "no", "wrong"]
        refs = ["yes", "no", "right"]
        acc = compute_accuracy(preds, refs)
        assert abs(acc - 0.6667) < 0.01

    def test_case_insensitive(self):
        preds = ["YES", "No"]
        refs = ["yes", "no"]
        assert compute_accuracy(preds, refs) == 1.0

    def test_empty(self):
        assert compute_accuracy([], []) == 0.0


class TestAllMetrics:
    def test_compute_all(self):
        predictions = ["the output text"]
        references = ["the output text"]
        losses = [1.0]

        results = compute_all_metrics(predictions, references, losses)
        assert "perplexity" in results
        assert "bleu" in results
        assert "rouge" in results
        assert "accuracy" in results

    def test_selective_metrics(self):
        predictions = ["hello"]
        references = ["hello"]
        results = compute_all_metrics(
            predictions, references, metrics=["accuracy", "bleu"]
        )
        assert "accuracy" in results
        assert "bleu" in results
        assert "perplexity" not in results

    def test_perplexity_needs_losses(self):
        predictions = ["hello"]
        references = ["hello"]
        results = compute_all_metrics(
            predictions, references, losses=None, metrics=["perplexity"]
        )
        assert "perplexity" not in results


class TestImprovement:
    def test_improvement_higher_is_better(self):
        base = {"bleu": 0.3, "accuracy": 0.5}
        finetuned = {"bleu": 0.45, "accuracy": 0.75}
        imp = compute_improvement(base, finetuned)
        assert imp["bleu"] == 50.0
        assert imp["accuracy"] == 50.0

    def test_improvement_perplexity_lower_is_better(self):
        base = {"perplexity": 100.0}
        finetuned = {"perplexity": 50.0}
        imp = compute_improvement(base, finetuned)
        assert imp["perplexity"] == 50.0  # 50% improvement

    def test_no_improvement(self):
        base = {"bleu": 0.5}
        finetuned = {"bleu": 0.5}
        imp = compute_improvement(base, finetuned)
        assert imp["bleu"] == 0.0

    def test_degradation(self):
        base = {"accuracy": 0.8}
        finetuned = {"accuracy": 0.6}
        imp = compute_improvement(base, finetuned)
        assert imp["accuracy"] == -25.0
