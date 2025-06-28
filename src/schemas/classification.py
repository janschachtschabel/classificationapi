"""Classification request and response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassificationMode(str, Enum):
    """Classification mode enumeration."""

    SKOS = "skos"
    CUSTOM = "custom"


class ClassificationRequest(BaseModel):
    """
    Request model for text classification.

    Supports multiple classification approaches:
    - SKOS vocabulary-based classification
    - Custom category classification
    - Descriptive metadata field generation
    - Resource suggestion generation

    At least one of 'mode', 'generate_descriptive_fields', or 'resource_suggestion' must be specified.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    text: str = Field(
        min_length=1, max_length=50000, description="Text content to classify and analyze (1-50,000 characters)"
    )

    mode: ClassificationMode | None = Field(
        None,
        description="Classification approach: 'skos' for SKOS vocabularies, 'custom' for user-defined categories. Optional if only generating descriptive fields.",
    )

    vocabulary_sources: list[str] | None = Field(
        None,
        description="List of SKOS vocabulary URLs for classification (required for 'skos' mode unless only generating descriptive fields). Example: ['https://example.com/vocab.rdf']",
    )

    custom_categories: dict[str, list[str]] | None = Field(
        None,
        description="Custom classification categories as key-value pairs (required for 'custom' mode). Example: {'topic': ['science', 'technology'], 'sentiment': ['positive', 'negative']}",
    )

    model: str | None = Field(
        default="gpt-4.1-mini",
        description="OpenAI model to use for classification. Recommended: 'gpt-4.1-mini' for cost-effectiveness, 'gpt-4' for highest accuracy.",
    )

    temperature: float | None = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Controls randomness in model responses (0.0-2.0). Lower values (0.1-0.3) for consistent results, higher values (0.7-1.0) for creative responses.",
    )

    max_tokens: int | None = Field(
        default=15000,
        ge=1,
        le=50000,
        description="Maximum tokens in model response (1-50,000). Adjust based on expected response length and cost considerations.",
    )

    generate_descriptive_fields: bool = Field(
        default=False,
        description="Generate structured metadata fields (title, short_title, keywords, description) from the input text. Can be used alone or combined with classification modes.",
    )

    resource_suggestion: bool = Field(
        default=False,
        description="Generate resource suggestions for educational content databases based on teaching/learning contexts. Uses specialized German prompts for focus analysis and filter generation. Can be used alone or combined with other modes.",
    )

    # Cross-Encoder semantic filtering settings
    enable_semantic_filtering: bool = Field(
        default=True,
        description="Enable Cross-Encoder based semantic filtering to reduce vocabulary concepts before LLM processing",
    )

    max_concepts_per_vocabulary: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Maximum number of concepts per vocabulary. If exceeded, Cross-Encoder will select the most relevant concepts.",
    )

    @model_validator(mode="after")
    def validate_request(self) -> "ClassificationRequest":
        """Validate mode-specific requirements."""
        # Check if at least one processing option is enabled
        if not self.mode and not self.generate_descriptive_fields and not self.resource_suggestion:
            raise ValueError(
                "At least one of 'mode', 'generate_descriptive_fields', or 'resource_suggestion' must be specified"
            )

        # If only descriptive fields or resource suggestion are requested, no mode validation needed
        if (self.generate_descriptive_fields or self.resource_suggestion) and not self.mode:
            return self

        # If mode is specified, validate mode-specific requirements
        if self.mode == ClassificationMode.SKOS:
            # Allow empty vocabulary_sources if descriptive fields or resource suggestion are requested
            if (
                not self.generate_descriptive_fields
                and not self.resource_suggestion
                and (not self.vocabulary_sources or len(self.vocabulary_sources) == 0)
            ):
                raise ValueError(
                    "vocabulary_sources is required for 'skos' mode unless generate_descriptive_fields or resource_suggestion is true"
                )
        elif self.mode == ClassificationMode.CUSTOM:
            # Allow empty custom_categories if descriptive fields or resource suggestion are requested
            if (
                not self.generate_descriptive_fields
                and not self.resource_suggestion
                and (not self.custom_categories or len(self.custom_categories) == 0)
            ):
                raise ValueError(
                    "custom_categories is required for 'custom' mode unless generate_descriptive_fields or resource_suggestion is true"
                )
            # Validate that all categories have at least one value (only if categories are provided)
            if self.custom_categories:
                for key, values in self.custom_categories.items():
                    if not values or len(values) == 0:
                        raise ValueError(f"Category '{key}' must have at least one value")

        return self


class ClassificationMatch(BaseModel):
    """A single classification match."""

    id: str | None = Field(description="Unique identifier for the match (URL for SKOS, generated for custom)")

    label: str = Field(description="Human-readable label for the match")

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

    explanation: str = Field(description="Explanation for why this match was selected")


class DescriptiveFields(BaseModel):
    """
    Descriptive metadata fields generated from text content.

    These fields provide structured metadata that can be used for:
    - Content management and cataloging
    - Search engine optimization
    - Document summarization
    - Content discovery and recommendation
    """

    title: str = Field(
        max_length=50, description="Concise, descriptive main title capturing the core topic (maximum 50 characters)"
    )

    short_title: str = Field(
        max_length=25, description="Abbreviated version of the title for compact displays (maximum 25 characters)"
    )

    keywords: list[str] = Field(description="List of relevant keywords and key terms extracted from the text content")

    description: str = Field(
        min_length=200,
        max_length=500,
        description="Comprehensive description summarizing the main themes and key information (200-500 characters)",
    )


class ResourceSuggestionFilterSuggestion(BaseModel):
    """Individual filter suggestion for resource suggestions."""

    vocabulary_url: str = Field(description="Source vocabulary URL")

    vocabulary_name: str | None = Field(None, description="Human-readable vocabulary name")

    values: list[str] = Field(description="Suggested filter values (concept IDs)")

    uris: list[str] = Field(default_factory=list, description="Full URIs of the suggested concepts")

    labels: list[str] = Field(description="Human-readable labels for the values")

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

    reasoning: str = Field(description="Explanation for the suggestion")


class ResourceSuggestionFields(BaseModel):
    """
    Resource suggestion fields generated from teaching/learning context analysis.

    These fields provide intelligent suggestions for educational content database searches:
    - Focus analysis (content-based vs methodical approaches)
    - Learning phase identification
    - SKOS vocabulary-based filter suggestions
    - Optimized search terms and keywords
    """

    focus_type: str = Field(
        description="Primary focus: 'content-based', 'competency-based', or 'methodical'",
        pattern="^(content-based|competency-based|methodical)$",
    )

    focus_explanation: str = Field(description="Explanation of why this focus was determined")

    learning_phase: str | None = Field(
        None, description="Identified learning phase (e.g., 'introduction', 'elaboration', 'practice', 'assessment')"
    )

    phase_explanation: str | None = Field(None, description="Explanation of the identified learning phase")

    filter_suggestions: list[ResourceSuggestionFilterSuggestion] = Field(
        default_factory=list, description="Generated filter suggestions based on provided vocabularies"
    )

    search_term: str = Field(description="Primary search term for database query")

    keywords: list[str] = Field(default_factory=list, description="Additional keywords for search refinement")

    title: str = Field(max_length=100, description="Descriptive title for the resource need")

    description: str = Field(max_length=500, description="Detailed description of the resource requirement")


class ClassificationResult(BaseModel):
    """Classification result for a single property."""

    property: str = Field(description="Name of the classified property")

    matches: list[ClassificationMatch] = Field(description="List of matches for this property")


class ClassificationMetadata(BaseModel):
    """Metadata about the classification request."""

    model: str = Field(description="OpenAI model used for classification")

    model_settings: dict[str, float | int] = Field(description="Model settings used (temperature, max_tokens, etc.)")

    timestamp: datetime = Field(description="Timestamp when classification was completed")

    processing_time_ms: int = Field(description="Processing time in milliseconds")

    skos_concepts_found: int | None = Field(
        None, description="Total number of concepts found in SKOS vocabularies (only for SKOS mode)"
    )

    skos_concepts_processed: int | None = Field(
        None, description="Number of concepts actually processed after limits applied (only for SKOS mode)"
    )

    semantic_filtering_used: bool | None = Field(
        None,
        description="Whether Cross-Encoder semantic filtering was used for vocabulary filtering (only for SKOS mode)",
    )

    semantic_filtering_disabled: bool | None = Field(
        None, description="Whether Cross-Encoder semantic filtering was manually disabled (only for SKOS mode)"
    )

    semantic_filtering_info: dict[str, Any] | None = Field(
        None,
        description="Details about semantic filtering: max_concepts_per_vocabulary, vocabularies_filtered, reduction_percent (only when semantic filtering is used)",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ClassificationResponse(BaseModel):
    """
    Response model for text classification requests.

    Contains classification results, optional descriptive fields, optional resource suggestion fields, and processing metadata.
    The response structure adapts based on the requested classification modes and options.
    """

    classification_id: UUID = Field(
        description="Unique identifier for this classification request, useful for tracking and logging"
    )

    status: str = Field(
        description="Processing status of the classification request (typically 'completed' for successful requests)"
    )

    results: list[ClassificationResult] = Field(
        description="Classification results for each property/category. Empty list if only descriptive fields or resource suggestions were requested."
    )

    descriptive_fields: DescriptiveFields | None = Field(
        None,
        description="Generated descriptive metadata fields (title, short_title, keywords, description). Only present if generate_descriptive_fields=true.",
    )

    resource_suggestion_fields: ResourceSuggestionFields | None = Field(
        None,
        description="Generated resource suggestion fields (focus analysis, filter suggestions, search terms). Only present if resource_suggestion=true.",
    )

    metadata: ClassificationMetadata = Field(
        description="Processing metadata including model used, settings, timestamp, and performance metrics"
    )
