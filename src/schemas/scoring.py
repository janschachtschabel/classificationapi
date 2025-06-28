"""
Pydantic schemas for text scoring and evaluation functionality.
"""

from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator


class ScaleType(str, Enum):
    """Types of scoring scales."""

    BINARY = "binary"
    LIKERT_3 = "likert_3"
    LIKERT_4 = "likert_4"
    LIKERT_5 = "likert_5"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"


class EvaluationCriterion(BaseModel):
    """Single evaluation criterion."""

    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="Detailed description of the criterion")
    weight: float = Field(default=1.0, ge=0.0, le=5.0, description="Weight of this criterion (0.0-5.0)")
    scale_min: int | None = Field(default=None, description="Minimum value for custom scales")
    scale_max: int | None = Field(default=None, description="Maximum value for custom scales")


class EvaluationScale(BaseModel):
    """Scoring scale configuration."""

    min: int = Field(..., description="Minimum score value")
    max: int = Field(..., description="Maximum score value")
    type: ScaleType = Field(..., description="Type of scale")
    labels: dict[str, str] | None = Field(
        default=None,
        description="Optional labels for scale values (keys should be numeric strings like '1', '2', etc.)",
        examples=[{"1": "Poor", "2": "Fair", "3": "Good", "4": "Very Good", "5": "Excellent"}],
    )


class PredefinedMetric(BaseModel):
    """Predefined evaluation metric loaded from YAML."""

    name: str = Field(..., description="Name of the metric")
    description: str = Field(..., description="Description of what this metric evaluates")
    scale: EvaluationScale = Field(..., description="Scoring scale for this metric")
    criteria: list[EvaluationCriterion] = Field(..., description="List of evaluation criteria")


class CustomMetric(BaseModel):
    """Custom evaluation metric defined by the user."""

    name: str = Field(..., description="Name of the custom metric")
    description: str | None = Field(default=None, description="Description of the custom metric")
    scale: EvaluationScale = Field(..., description="Scoring scale")
    criteria: list[EvaluationCriterion] = Field(..., description="List of evaluation criteria")


class CriterionScore(BaseModel):
    """Score for a single criterion."""

    name: str = Field(..., description="Name of the criterion")
    score: int | float | str = Field(..., description="Score value (numeric or 'N/A')")
    max_score: int | float | str = Field(..., description="Maximum possible score (numeric or 'N/A')")
    reasoning: str = Field(..., description="Explanation for this score")
    weight: float = Field(..., description="Weight of this criterion")


class ScoringResult(BaseModel):
    """Result of text scoring evaluation."""

    metric_name: str = Field(..., description="Name of the evaluated metric")
    overall_score: float = Field(..., description="Overall weighted score")
    max_possible_score: float = Field(..., description="Maximum possible weighted score")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0.0-1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the evaluation")
    criterion_scores: list[CriterionScore] = Field(..., description="Scores for individual criteria")
    overall_reasoning: str = Field(..., description="Overall reasoning for the evaluation")
    suggested_improvements: str | None = Field(default=None, description="Suggested text improvements")


class ScoringRequest(BaseModel):
    """Request for text scoring evaluation."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to be evaluated")

    # Predefined metrics
    predefined_metrics: list[str] | None = Field(
        default=None,
        description="List of predefined metrics to apply. Use 'custom' to enable custom metrics processing.",
    )

    # Custom metrics
    custom_metrics: list[CustomMetric] | None = Field(default=None, description="List of custom evaluation metrics")

    # Additional options
    include_improvements: bool = Field(default=False, description="Whether to include suggested text improvements")

    language: str = Field(default="de", description="Language for evaluation (de, en, fr)")

    @model_validator(mode="after")
    def validate_metrics(self) -> "ScoringRequest":
        """Validate that at least one metric is specified."""
        # Check if we have predefined metrics (excluding "custom" flag)
        actual_predefined: list[str] = []
        custom_flag_present = False

        if self.predefined_metrics:
            for metric in self.predefined_metrics:
                if metric == "custom":
                    custom_flag_present = True
                else:
                    actual_predefined.append(metric)

        # Filter out dummy/default custom metrics that Swagger UI might send
        valid_custom_metrics: list[CustomMetric] = []
        if self.custom_metrics:
            # Use list comprehension to filter out dummy metrics
            valid_custom_metrics = [
                metric for metric in self.custom_metrics
                if not (
                    metric.name == "string"
                    and metric.description == "string"
                    and all(c.name == "string" and c.description == "string" for c in metric.criteria)
                )
            ]

        # Update the custom_metrics list to only include valid ones
        self.custom_metrics = valid_custom_metrics if valid_custom_metrics else None

        # We need either:
        # 1. At least one actual predefined metric, OR
        # 2. Valid custom metrics provided (with or without "custom" flag)
        has_predefined = len(actual_predefined) > 0
        has_custom = self.custom_metrics and len(self.custom_metrics) > 0

        if not has_predefined and not has_custom:
            raise ValueError("At least one predefined metric or custom metric must be specified")

        # If valid custom metrics are provided without "custom" flag, automatically add it
        if has_custom and not custom_flag_present:
            if not self.predefined_metrics:
                self.predefined_metrics = ["custom"]
            else:
                self.predefined_metrics.append("custom")

        # If no valid custom metrics but "custom" flag is present, remove the flag
        if not has_custom and custom_flag_present and self.predefined_metrics:
            self.predefined_metrics = [m for m in self.predefined_metrics if m != "custom"]

        return self


class ScoringResponse(BaseModel):
    """Response for text scoring evaluation."""

    text: str = Field(..., description="Original text that was evaluated")
    results: list[ScoringResult] = Field(..., description="Scoring results for each metric")
    processing_time: float = Field(..., description="Processing time in seconds")
    language: str = Field(..., description="Language used for evaluation")

    # Metadata
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class ScoringMetadata(BaseModel):
    """Metadata for scoring operations."""

    total_metrics: int = Field(..., description="Total number of metrics evaluated")
    predefined_metrics_used: list[str] = Field(default_factory=list, description="Names of predefined metrics used")
    custom_metrics_used: list[str] = Field(default_factory=list, description="Names of custom metrics used")
    openai_calls: int = Field(default=0, description="Number of OpenAI API calls made")
    total_tokens_used: int | None = Field(default=None, description="Total tokens used in OpenAI calls")
