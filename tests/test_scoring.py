"""
Tests for scoring functionality.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.schemas.scoring import (
    CustomMetric,
    EvaluationCriterion,
    EvaluationScale,
    ScaleType,
    ScoringRequest,
    ScoringResponse,
)
from src.services.scoring_service import ScoringService


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_default_model = "gpt-4.1-mini"
    settings.openai_temperature = 0.2
    settings.openai_max_tokens = 1000
    settings.openai_timeout = 30
    settings.default_temperature = 0.2
    settings.default_max_tokens = 15000
    settings.openai_timeout_seconds = 60
    return settings


@pytest.fixture
def scoring_service(mock_settings):
    """Create scoring service for testing."""
    with patch("src.services.scoring_service.Path.exists", return_value=False):
        service = ScoringService(mock_settings)
    return service


@pytest.fixture
def sample_custom_metric():
    """Sample custom metric for testing."""
    return CustomMetric(
        name="Test Metric",
        description="A test metric",
        scale=EvaluationScale(min=1, max=5, type=ScaleType.LIKERT_5),
        criteria=[EvaluationCriterion(name="Test Criterion", description="A test criterion", weight=1.0)],
    )


@pytest.fixture
def sample_scoring_request(sample_custom_metric):
    """Sample scoring request for testing."""
    return ScoringRequest(text="This is a test text for evaluation.", custom_metrics=[sample_custom_metric])


class TestScoringService:
    """Test cases for ScoringService."""

    @pytest.mark.asyncio
    async def test_score_text_with_custom_metric(self, scoring_service, sample_scoring_request):
        """Test scoring text with custom metric."""
        # Mock OpenAI response
        mock_response = {
            "criterion_scores": [
                {"name": "Test Criterion", "score": 4, "max_score": 5, "reasoning": "Good quality text", "weight": 1.0}
            ],
            "overall_reasoning": "Overall good quality",
            "confidence": 0.85,
            "suggested_improvements": "Minor improvements needed",
        }

        with patch.object(scoring_service, "_call_openai", return_value=json.dumps(mock_response)):
            result = await scoring_service.score_text(sample_scoring_request)

            assert isinstance(result, ScoringResponse)
            assert result.text == sample_scoring_request.text
            assert len(result.results) == 1
            assert result.results[0].metric_name == "Test Metric"
            assert result.results[0].overall_score == 4.0
            assert result.results[0].normalized_score == 0.8
            assert result.results[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_score_text_with_predefined_metric(self, scoring_service):
        """Test scoring text with predefined metric."""
        # Mock predefined metric
        mock_yaml_data = {
            "evaluation_criteria": {
                "name": "Sachrichtigkeit",
                "description": "Test description",
                "scale": {"min": 0, "max": 1, "type": "binary"},
                "criteria": [{"name": "Test Criterion", "description": "Test description", "weight": 1.0}],
            },
            "prompt_template": "Test prompt: {criteria_text} {text} {output_format}",
        }

        # Mock file operations
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("sachrichtigkeit.yaml")]),
            patch("builtins.open", mock_open(read_data="test")),
            patch("yaml.safe_load", return_value=mock_yaml_data),
        ):
            # Reload metrics
            scoring_service._load_predefined_metrics()

            # Create request with predefined metric
            request = ScoringRequest(text="Test text", predefined_metrics=["sachrichtigkeit"])

            # Mock OpenAI response
            mock_response = {
                "criterion_scores": [
                    {"name": "Test Criterion", "score": 1, "max_score": 1, "reasoning": "Meets criteria", "weight": 1.0}
                ],
                "overall_reasoning": "Good quality",
                "confidence": 0.9,
                "suggested_improvements": None,
            }

            with patch.object(scoring_service, "_call_openai", return_value=json.dumps(mock_response)):
                result = await scoring_service.score_text(request)

                assert len(result.results) == 1
                assert result.results[0].metric_name == "Sachrichtigkeit"
                assert result.results[0].normalized_score == 1.0

    def test_build_criteria_text(self, scoring_service):
        """Test building criteria text for prompt."""
        criteria = [
            EvaluationCriterion(name="Criterion 1", description="Description 1", weight=1.0),
            EvaluationCriterion(name="Criterion 2", description="Description 2", weight=2.0),
        ]

        result = scoring_service._build_criteria_text(criteria)

        assert "**Criterion 1**:" in result
        assert "Description 1" in result
        assert "**Criterion 2** (Gewichtung: 2.0):" in result
        assert "Description 2" in result

    def test_build_output_format(self, scoring_service):
        """Test building output format for prompt."""
        criteria = [EvaluationCriterion(name="Test Criterion", description="Test description", weight=1.0)]
        scale = EvaluationScale(min=1, max=5, type=ScaleType.LIKERT_5)

        result = scoring_service._build_output_format(criteria, scale)

        assert "JSON-Format" in result
        assert "Test Criterion" in result
        assert "zwischen 1 und 5" in result
        assert "confidence" in result
        assert "suggested_improvements" in result

    @pytest.mark.asyncio
    async def test_call_openai(self, scoring_service):
        """Test OpenAI API call."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        with patch.object(
            scoring_service.openai_client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await scoring_service._call_openai("Test prompt")

            assert result == "Test response"

    def test_parse_evaluation_response_success(self, scoring_service, sample_custom_metric):
        """Test successful parsing of evaluation response."""
        response = json.dumps(
            {
                "criterion_scores": [
                    {"name": "Test Criterion", "score": 4, "max_score": 5, "reasoning": "Good quality", "weight": 1.0}
                ],
                "overall_reasoning": "Overall assessment",
                "confidence": 0.8,
                "suggested_improvements": "Some improvements",
            }
        )

        # Convert custom metric to predefined for testing
        predefined_metric = scoring_service._convert_custom_to_predefined(sample_custom_metric)

        result = scoring_service._parse_evaluation_response(response, predefined_metric, include_improvements=True)

        assert result.metric_name == "Test Metric"
        assert result.overall_score == 4.0
        assert result.normalized_score == 0.8
        assert result.confidence == 0.8
        assert len(result.criterion_scores) == 1

    def test_parse_evaluation_response_error(self, scoring_service, sample_custom_metric):
        """Test parsing of malformed evaluation response."""
        response = "Invalid JSON response"

        predefined_metric = scoring_service._convert_custom_to_predefined(sample_custom_metric)

        result = scoring_service._parse_evaluation_response(response, predefined_metric, include_improvements=False)

        assert result.metric_name == "Test Metric"
        assert result.overall_score == 0.0
        assert result.normalized_score == 0.0
        assert result.confidence == 0.0
        assert "Parsing error" in result.overall_reasoning

    def test_get_available_metrics(self, scoring_service):
        """Test getting available metrics."""
        # Mock some metrics in cache
        scoring_service.criteria_cache = {
            "test1": MagicMock(description="Test 1"),
            "test2": MagicMock(description="Test 2"),
        }

        result = scoring_service.get_available_metrics()

        assert len(result) == 2
        assert "test1" in result
        assert "test2" in result


class TestScoringSchemas:
    """Test cases for scoring schemas."""

    def test_scoring_request_validation_success(self, sample_custom_metric):
        """Test successful validation of scoring request."""
        request = ScoringRequest(text="Test text", custom_metrics=[sample_custom_metric])

        assert request.text == "Test text"
        assert len(request.custom_metrics) == 1
        assert request.include_improvements is False  # Default is now False
        assert request.language == "de"

    def test_scoring_request_validation_no_metrics(self):
        """Test validation failure when no metrics specified."""
        with pytest.raises(ValidationError, match="At least one predefined metric or custom metric must be specified"):
            ScoringRequest(text="Test text")

    def test_scoring_request_validation_predefined_only(self):
        """Test validation success with predefined metrics only."""
        request = ScoringRequest(text="Test text", predefined_metrics=["sachrichtigkeit"])

        assert request.predefined_metrics == ["sachrichtigkeit"]
        assert request.custom_metrics is None

    def test_custom_metric_creation(self):
        """Test creation of custom metric."""
        metric = CustomMetric(
            name="Test Metric",
            scale=EvaluationScale(min=1, max=10, type=ScaleType.CUSTOM),
            criteria=[EvaluationCriterion(name="Criterion 1", description="Test criterion", weight=2.5)],
        )

        assert metric.name == "Test Metric"
        assert metric.scale.min == 1
        assert metric.scale.max == 10
        assert metric.scale.type == ScaleType.CUSTOM
        assert len(metric.criteria) == 1
        assert metric.criteria[0].weight == 2.5
