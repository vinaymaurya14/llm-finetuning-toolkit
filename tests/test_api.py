"""FastAPI endpoint tests using TestClient (CPU-only, mocked training)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "gpu" in data

    def test_health_has_gpu_info(self):
        response = client.get("/health")
        data = response.json()
        gpu = data["gpu"]
        assert "available" in gpu


class TestDatasetEndpoints:
    def test_list_datasets_empty(self):
        response = client.get("/datasets")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_prepare_dataset_invalid_source(self):
        response = client.post(
            "/datasets/prepare",
            json={
                "source": "/nonexistent/path/to/data.json",
                "source_format": "alpaca",
                "target_format": "alpaca",
            },
        )
        assert response.status_code in (404, 400, 500)

    def test_prepare_dataset_from_local_file(self, tmp_path):
        import app.config as cfg

        original_dir = cfg.settings.dataset_dir
        cfg.settings.dataset_dir = str(tmp_path / "datasets")

        # Create test data
        data = [
            {"instruction": f"Q{i} about a topic here", "input": "", "output": f"Answer {i} with content"}
            for i in range(10)
        ]
        source_file = tmp_path / "api_test.jsonl"
        with open(source_file, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        response = client.post(
            "/datasets/prepare",
            json={
                "source": str(source_file),
                "source_format": "alpaca",
                "target_format": "alpaca",
                "split_ratio": 0.8,
                "name": "api_test_ds",
                "filters": {"min_length": 5, "max_length": 5000},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "api_test_ds"
        assert data["num_samples"] == 10

        cfg.settings.dataset_dir = original_dir


class TestTrainingEndpoints:
    def test_list_jobs_empty(self):
        response = client.get("/training/jobs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_job(self):
        response = client.get("/training/status/nonexistent_job")
        assert response.status_code == 404

    @patch("app.services.trainer.detect_gpu")
    def test_sft_no_gpu(self, mock_gpu):
        from app.utils.gpu_utils import GPUStatus
        mock_gpu.return_value = GPUStatus(available=False)

        response = client.post(
            "/training/sft",
            json={
                "dataset_name": "test_dataset",
                "base_model": "test-model-7b",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "GPU" in data.get("error_message", "") or "No GPU" in data.get("error_message", "")

    @patch("app.services.trainer.detect_gpu")
    def test_dpo_no_gpu(self, mock_gpu):
        from app.utils.gpu_utils import GPUStatus
        mock_gpu.return_value = GPUStatus(available=False)

        response = client.post(
            "/training/dpo",
            json={
                "dataset_name": "test_dataset",
                "base_model": "test-model-7b",
                "beta": 0.1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"


class TestModelRegistryEndpoints:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_model(self):
        response = client.get("/models/nonexistent_adapter")
        assert response.status_code == 404

    def test_delete_nonexistent_model(self):
        response = client.delete("/models/nonexistent_adapter")
        assert response.status_code == 404


class TestInferenceEndpoints:
    def test_unload_models(self):
        response = client.post("/inference/unload")
        assert response.status_code == 200
        assert "unloaded" in response.json()["message"].lower()

    def test_generate_missing_deps(self):
        """Test that generation gracefully handles missing model/dependencies."""
        response = client.post(
            "/inference/generate",
            json={
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
            },
        )
        # Should return 500 or 503 since no model is loadable in test env
        assert response.status_code in (500, 503)


class TestEvaluationEndpoints:
    def test_evaluate_missing_deps(self):
        """Test that evaluation gracefully handles missing adapter."""
        response = client.post(
            "/evaluate",
            json={
                "adapter_path": "/nonexistent/adapter",
                "num_samples": 5,
            },
        )
        # Should either succeed with error info or return 500
        assert response.status_code in (200, 500)


class TestRequestValidation:
    def test_generate_empty_prompt(self):
        response = client.post(
            "/inference/generate",
            json={
                "prompt": "",
                "max_tokens": 50,
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_sft_missing_dataset_name(self):
        response = client.post(
            "/training/sft",
            json={},
        )
        assert response.status_code == 422

    def test_invalid_split_ratio(self):
        response = client.post(
            "/datasets/prepare",
            json={
                "source": "test",
                "split_ratio": 1.5,  # Invalid: > 0.99
            },
        )
        assert response.status_code == 422
