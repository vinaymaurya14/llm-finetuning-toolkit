"""Tests for dataset engine: format conversion, quality filters, splitting."""

import json
import tempfile
from pathlib import Path

import pytest

from app.core.dataset_formats import (
    convert_format,
    detect_format,
    get_text_for_filtering,
    to_alpaca,
    to_conversational,
    to_sharegpt,
)
from app.core.quality_filters import QualityFilter, compute_quality_score


# ────────────────────── Format Detection ──────────────────────


class TestFormatDetection:
    def test_detect_alpaca(self):
        sample = {"instruction": "Summarize this", "input": "...", "output": "Summary"}
        assert detect_format(sample) == "alpaca"

    def test_detect_sharegpt(self):
        sample = {"conversations": [{"from": "human", "value": "Hi"}]}
        assert detect_format(sample) == "sharegpt"

    def test_detect_conversational(self):
        sample = {"messages": [{"role": "user", "content": "Hi"}]}
        assert detect_format(sample) == "conversational"

    def test_detect_dpo(self):
        sample = {"prompt": "Q", "chosen": "Good A", "rejected": "Bad A"}
        assert detect_format(sample) == "dpo"

    def test_detect_qa_as_alpaca(self):
        sample = {"question": "What?", "answer": "This"}
        assert detect_format(sample) == "alpaca"

    def test_detect_unknown_defaults_alpaca(self):
        sample = {"foo": "bar"}
        assert detect_format(sample) == "alpaca"


# ────────────────────── Format Conversion ──────────────────────


class TestFormatConversion:
    def test_alpaca_to_alpaca(self):
        row = {"instruction": "Test", "input": "data", "output": "result"}
        result = convert_format(row, "alpaca", "alpaca")
        assert result["instruction"] == "Test"
        assert result["output"] == "result"

    def test_alpaca_to_sharegpt(self):
        row = {"instruction": "Explain X", "input": "", "output": "X is..."}
        result = convert_format(row, "alpaca", "sharegpt")
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_alpaca_to_conversational(self):
        row = {"instruction": "Do Y", "input": "", "output": "Done Y"}
        result = convert_format(row, "alpaca", "conversational")
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_sharegpt_to_alpaca(self):
        row = {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "AI is..."},
            ]
        }
        result = convert_format(row, "sharegpt", "alpaca")
        assert result["instruction"] == "What is AI?"
        assert result["output"] == "AI is..."

    def test_sharegpt_to_conversational(self):
        row = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
            ]
        }
        result = convert_format(row, "sharegpt", "conversational")
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_conversational_to_alpaca(self):
        row = {
            "messages": [
                {"role": "user", "content": "Help me"},
                {"role": "assistant", "content": "Sure"},
            ]
        }
        result = convert_format(row, "conversational", "alpaca")
        assert result["instruction"] == "Help me"
        assert result["output"] == "Sure"

    def test_same_format_passthrough(self):
        row = {"instruction": "X", "input": "", "output": "Y"}
        result = convert_format(row, "alpaca", "alpaca")
        assert result == {"instruction": "X", "input": "", "output": "Y"}

    def test_dpo_to_dpo(self):
        row = {"prompt": "Q", "chosen": "Good", "rejected": "Bad"}
        result = convert_format(row, "dpo", "dpo")
        assert result["prompt"] == "Q"

    def test_invalid_to_dpo_raises(self):
        row = {"instruction": "X", "input": "", "output": "Y"}
        with pytest.raises(ValueError, match="Cannot auto-convert"):
            convert_format(row, "alpaca", "dpo")


# ────────────────────── Text Extraction ──────────────────────


class TestTextExtraction:
    def test_alpaca_text(self):
        row = {"instruction": "Do X", "input": "with Y", "output": "Result Z"}
        text = get_text_for_filtering(row, "alpaca")
        assert "Do X" in text
        assert "Result Z" in text

    def test_sharegpt_text(self):
        row = {"conversations": [{"value": "Hello"}, {"value": "World"}]}
        text = get_text_for_filtering(row, "sharegpt")
        assert "Hello" in text
        assert "World" in text

    def test_dpo_text(self):
        row = {"prompt": "Q", "chosen": "A", "rejected": "B"}
        text = get_text_for_filtering(row, "dpo")
        assert "Q" in text
        assert "A" in text


# ────────────────────── Quality Filters ──────────────────────


class TestQualityFilters:
    def test_length_filter_too_short(self):
        qf = QualityFilter(min_length=20)
        assert not qf.passes("Hi")
        assert qf.stats.filtered_short == 1

    def test_length_filter_too_long(self):
        qf = QualityFilter(max_length=10)
        assert not qf.passes("A" * 100)
        assert qf.stats.filtered_long == 1

    def test_length_filter_passes(self):
        qf = QualityFilter(min_length=5, max_length=100)
        assert qf.passes("This is a valid text string.")

    def test_duplicate_filter(self):
        qf = QualityFilter(remove_duplicates=True, min_length=1)
        assert qf.passes("unique text one")
        assert qf.passes("unique text two")
        assert not qf.passes("unique text one")
        assert qf.stats.filtered_duplicate == 1

    def test_no_duplicate_filter(self):
        qf = QualityFilter(remove_duplicates=False, min_length=1)
        assert qf.passes("same text")
        assert qf.passes("same text")

    def test_batch_filter(self):
        qf = QualityFilter(min_length=5, max_length=200, remove_duplicates=True)
        texts = [
            "Short",  # too short (depends on threshold, but 5 chars = passes)
            "Hi",  # too short
            "This is a good sample with enough content.",
            "This is a good sample with enough content.",  # duplicate
            "Another valid piece of text here.",
        ]
        results = qf.filter_batch(texts)
        indices = [idx for idx, _ in results]
        assert 0 in indices  # "Short" has 5 chars, passes min_length=5
        assert 1 not in indices  # "Hi" has 2 chars, filtered
        assert 2 in indices
        assert 3 not in indices  # duplicate
        assert 4 in indices

    def test_quality_score_good_text(self):
        score = compute_quality_score(
            "This is a well-written sentence with proper grammar and structure."
        )
        assert score >= 0.5

    def test_quality_score_bad_text(self):
        score = compute_quality_score("x")
        assert score < 0.8

    def test_quality_score_repetitive(self):
        score = compute_quality_score("bad bad bad bad bad bad bad bad bad bad")
        assert score < 0.7


# ────────────────────── Dataset File Loading ──────────────────────


class TestDatasetFileLoading:
    def test_load_jsonl(self):
        from app.services.dataset_engine import _load_from_file

        data = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "", "output": "A2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            f.flush()

            loaded = _load_from_file(f.name)
            assert len(loaded) == 2
            assert loaded[0]["instruction"] == "Q1"

        Path(f.name).unlink()

    def test_load_json(self):
        from app.services.dataset_engine import _load_from_file

        data = [
            {"instruction": "Q1", "input": "", "output": "A1"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            loaded = _load_from_file(f.name)
            assert len(loaded) == 1

        Path(f.name).unlink()

    def test_load_csv(self):
        from app.services.dataset_engine import _load_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("instruction,input,output\n")
            f.write("Q1,,A1\n")
            f.write("Q2,,A2\n")
            f.flush()

            loaded = _load_from_file(f.name)
            assert len(loaded) == 2

        Path(f.name).unlink()

    def test_load_nonexistent_raises(self):
        from app.services.dataset_engine import _load_from_file

        with pytest.raises(FileNotFoundError):
            _load_from_file("/nonexistent/path.json")

    def test_load_unsupported_format_raises(self):
        from app.services.dataset_engine import _load_from_file

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported"):
                _load_from_file(f.name)

        Path(f.name).unlink()

    def test_max_samples(self):
        from app.services.dataset_engine import _load_from_file

        data = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(100)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            f.flush()

            loaded = _load_from_file(f.name, max_samples=10)
            assert len(loaded) == 10

        Path(f.name).unlink()


# ────────────────────── Full Pipeline ──────────────────────


class TestFullPipeline:
    def test_prepare_from_local_file(self, tmp_path):
        from app.services.dataset_engine import prepare_dataset
        from app.models.schemas import DatasetFormat, QualityFilterConfig
        import app.config as cfg

        # Override dataset dir for test
        original_dir = cfg.settings.dataset_dir
        cfg.settings.dataset_dir = str(tmp_path / "datasets")

        data = [
            {"instruction": f"Question number {i} about a topic", "input": "", "output": f"Answer {i} with details"}
            for i in range(20)
        ]
        source_file = tmp_path / "test_data.jsonl"
        with open(source_file, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = prepare_dataset(
            source=str(source_file),
            source_format=DatasetFormat.ALPACA,
            target_format=DatasetFormat.ALPACA,
            split_ratio=0.8,
            filters=QualityFilterConfig(min_length=5, max_length=5000),
            name="test_dataset",
        )

        assert result.name == "test_dataset"
        assert result.num_samples == 20
        assert result.split_sizes["train"] == 16
        assert result.split_sizes["val"] == 4
        assert result.format == DatasetFormat.ALPACA

        # Restore
        cfg.settings.dataset_dir = original_dir
