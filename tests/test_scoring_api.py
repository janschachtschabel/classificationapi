"""
Tests for scoring API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_scoring_service
from src.main import create_app
from src.schemas.scoring import CriterionScore, ScoringResponse, ScoringResult


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_scoring_service():
    """Create mock scoring service."""
    return AsyncMock()


@pytest.fixture
def sample_scoring_result():
    """Sample scoring result for testing."""
    return ScoringResult(
        metric_name="Test Metric",
        overall_score=4.0,
        max_possible_score=5.0,
        normalized_score=0.8,
        confidence=0.85,
        criterion_scores=[
            CriterionScore(name="Test Criterion", score=4, max_score=5, reasoning="Good quality text", weight=1.0)
        ],
        overall_reasoning="Overall good quality",
        suggested_improvements="Minor improvements needed",
    )


@pytest.fixture
def sample_scoring_response(sample_scoring_result):
    """Sample scoring response for testing."""
    return ScoringResponse(
        text="Test text for evaluation",
        results=[sample_scoring_result],
        processing_time=1.23,
        language="de",
        metadata={
            "total_metrics": 1,
            "predefined_metrics_used": [],
            "custom_metrics_used": ["Test Metric"],
            "openai_model_used": "gpt-4.1-mini",
        },
    )


class TestScoringEndpoints:
    """Test cases for scoring API endpoints."""

    def test_evaluate_text_success_custom_metric(self, client, mock_scoring_service, sample_scoring_response):
        """Test successful text evaluation with custom metric."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        mock_scoring_service.score_text = AsyncMock(return_value=sample_scoring_response)

        request_data = {
            "text": "Test text for evaluation",
            "custom_metrics": [
                {
                    "name": "Test Metric",
                    "description": "A test metric",
                    "scale": {"min": 1, "max": 5, "type": "likert_5"},
                    "criteria": [{"name": "Test Criterion", "description": "A test criterion", "weight": 1.0}],
                }
            ],
        }

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test text for evaluation"
        assert len(data["results"]) == 1
        assert data["results"][0]["metric_name"] == "Test Metric"
        assert data["results"][0]["normalized_score"] == 0.8
        assert data["processing_time"] == 1.23

        mock_scoring_service.score_text.assert_called_once()

    def test_evaluate_text_success_predefined_metric(self, client, mock_scoring_service, sample_scoring_response):
        """Test successful text evaluation with predefined metric."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        mock_scoring_service.score_text = AsyncMock(return_value=sample_scoring_response)

        request_data = {"text": "Test text for evaluation", "predefined_metrics": ["sachrichtigkeit"]}

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test text for evaluation"
        assert len(data["results"]) == 1

        mock_scoring_service.score_text.assert_called_once()

    def test_evaluate_text_success_combined_metrics(self, client, mock_scoring_service, sample_scoring_response):
        """Test successful text evaluation with both predefined and custom metrics."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        mock_scoring_service.score_text = AsyncMock(return_value=sample_scoring_response)

        request_data = {
            "text": "Test text for evaluation",
            "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],
            "custom_metrics": [
                {
                    "name": "Custom Metric",
                    "scale": {"min": 0, "max": 1, "type": "binary"},
                    "criteria": [{"name": "Custom Criterion", "description": "Custom test criterion", "weight": 1.0}],
                }
            ],
            "include_improvements": True,
            "language": "de",
        }

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "de"

        mock_scoring_service.score_text.assert_called_once()

    def test_evaluate_text_validation_error_no_metrics(self, client):
        """Test validation error when no metrics specified."""
        request_data = {"text": "Test text for evaluation"}

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_evaluate_text_validation_error_empty_text(self, client):
        """Test validation error with empty text."""
        request_data = {"text": "", "predefined_metrics": ["sachrichtigkeit"]}

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_evaluate_text_validation_error_invalid_predefined_metric(self, client):
        """Test that invalid predefined metrics are handled gracefully with warnings."""
        request_data = {"text": "Test text", "predefined_metrics": ["invalid_metric"]}

        response = client.post("/scoring/evaluate", json=request_data)

        # Invalid metrics are now handled gracefully with warnings, not errors
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_evaluate_text_validation_error_invalid_custom_metric(self, client):
        """Test validation error with invalid custom metric."""
        request_data = {
            "text": "Test text",
            "custom_metrics": [
                {"name": "Invalid Metric", "scale": {"min": 1, "max": 5, "type": "invalid_type"}, "criteria": []}
            ],
        }

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_evaluate_text_service_error(self, client, mock_scoring_service):
        """Test handling of service errors."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        # Create a simple mock that raises an exception
        async def mock_score_text(*args, **kwargs):
            raise Exception("Service error")

        mock_scoring_service.score_text = AsyncMock(side_effect=mock_score_text)

        request_data = {"text": "Test text", "predefined_metrics": ["sachrichtigkeit"]}

        response = client.post("/scoring/evaluate", json=request_data)

        # Check if we get a 500 status code (internal server error)
        assert response.status_code == 500
        # The response format might vary, so let's be more flexible
        data = response.json()
        assert "detail" in data or "error" in data

    def test_evaluate_text_value_error(self, client, mock_scoring_service):
        """Test handling of validation errors from service."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        # Create a simple mock that raises a ValueError
        async def mock_score_text(*args, **kwargs):
            raise ValueError("Invalid input")

        mock_scoring_service.score_text = AsyncMock(side_effect=mock_score_text)

        request_data = {"text": "Test text", "predefined_metrics": ["sachrichtigkeit"]}

        response = client.post("/scoring/evaluate", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data or "error" in data

    def test_get_available_metrics_success(self, client, mock_scoring_service):
        """Test successful retrieval of available metrics."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        mock_scoring_service.get_available_metrics = MagicMock(
            return_value={
                "sachrichtigkeit": "Bewertung der sachlichen Korrektheit und Genauigkeit von Texten",
                "neutralitaet": "Bewertung der neutralen und objektiven Darstellung von Inhalten",
            }
        )

        response = client.get("/scoring/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "sachrichtigkeit" in data
        assert "neutralitaet" in data
        assert data["sachrichtigkeit"] == "Bewertung der sachlichen Korrektheit und Genauigkeit von Texten"

    def test_get_available_metrics_service_error(self, client, mock_scoring_service):
        """Test handling of service errors when getting metrics."""
        app = client.app
        app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service

        mock_scoring_service.get_available_metrics = MagicMock(side_effect=Exception("Service error"))

        response = client.get("/scoring/metrics")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data or "error" in data


class TestScoringRequestValidation:
    """Test cases for scoring request validation."""

    def test_valid_request_predefined_only(self, client):
        """Test valid request with predefined metrics only."""
        request_data = {"text": "Test text for evaluation", "predefined_metrics": ["sachrichtigkeit"]}

        # This should not raise validation error (will fail at service level in test)
        response = client.post("/scoring/evaluate", json=request_data)
        # We expect 500 because service is not mocked, but validation should pass
        assert response.status_code in [500, 200]  # 200 if service works, 500 if not mocked

    def test_valid_request_custom_only(self, client):
        """Test valid request with custom metrics only."""
        request_data = {
            "text": "Test text for evaluation",
            "custom_metrics": [
                {
                    "name": "Test Metric",
                    "scale": {"min": 1, "max": 5, "type": "likert_5"},
                    "criteria": [{"name": "Test Criterion", "description": "Test description", "weight": 1.0}],
                }
            ],
        }

        # This should not raise validation error (will fail at service level in test)
        response = client.post("/scoring/evaluate", json=request_data)
        # We expect 500 because service is not mocked, but validation should pass
        assert response.status_code in [500, 200]  # 200 if service works, 500 if not mocked

    def test_invalid_request_no_metrics(self, client):
        """Test invalid request with no metrics."""
        request_data = {"text": "Test text for evaluation"}

        response = client.post("/scoring/evaluate", json=request_data)
        assert response.status_code == 422

    def test_invalid_request_empty_text(self, client):
        """Test invalid request with empty text."""
        request_data = {"text": "", "predefined_metrics": ["sachrichtigkeit"]}

        response = client.post("/scoring/evaluate", json=request_data)
        assert response.status_code == 422

    def test_invalid_request_text_too_long(self, client):
        """Test invalid request with text too long."""
        request_data = {
            "text": "x" * 10001,  # Exceeds max length of 10000
            "predefined_metrics": ["sachrichtigkeit"],
        }

        response = client.post("/scoring/evaluate", json=request_data)
        assert response.status_code == 422
