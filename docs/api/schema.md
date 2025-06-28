# API Schema Reference

Complete reference for all API schemas, request/response models, and data structures used in the Metadata Classification API.

## Classification Schemas

### ClassificationRequest

Request schema for text classification operations.

```python
class ClassificationRequest(BaseModel):
    """Request model for text classification."""
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="Text content to classify"
    )
    
    mode: Optional[Literal["skos", "custom"]] = Field(
        None,
        description="Classification mode: 'skos' for SKOS vocabularies, 'custom' for user-defined categories"
    )
    
    vocabulary_sources: Optional[List[str]] = Field(
        None,
        description="SKOS vocabulary sources to use (for SKOS mode)"
    )
    
    categories: Optional[List[str]] = Field(
        None,
        description="Custom categories for classification (for custom mode)"
    )
    
    generate_descriptive_fields: bool = Field(
        False,
        description="Whether to generate descriptive metadata fields"
    )
    
    resource_suggestion: bool = Field(
        False,
        description="Whether to generate resource suggestions"
    )
    
    confidence_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for classifications"
    )
    
    max_classifications: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of classifications to return per vocabulary"
    )
```

**Example Request:**
```json
{
    "text": "Introduction to machine learning algorithms for educational purposes",
    "mode": "skos",
    "vocabulary_sources": ["discipline", "educational_context"],
    "generate_descriptive_fields": true,
    "resource_suggestion": true,
    "confidence_threshold": 0.3,
    "max_classifications": 5
}
```

### ClassificationResponse

Response schema for classification operations.

```python
class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    
    classification_id: str = Field(
        description="Unique identifier for this classification"
    )
    
    status: Literal["completed", "error"] = Field(
        description="Processing status"
    )
    
    results: Optional[List[VocabularyClassificationResult]] = Field(
        None,
        description="Classification results by vocabulary"
    )
    
    descriptive_fields: Optional[DescriptiveFields] = Field(
        None,
        description="Generated descriptive metadata"
    )
    
    resource_suggestion_fields: Optional[ResourceSuggestionFields] = Field(
        None,
        description="Resource suggestion metadata"
    )
    
    processing_time: float = Field(
        description="Processing time in seconds"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if status is 'error'"
    )
```

### VocabularyClassificationResult

Classification results for a specific vocabulary.

```python
class VocabularyClassificationResult(BaseModel):
    """Classification results for a specific vocabulary."""
    
    vocabulary_name: str = Field(
        description="Name of the vocabulary used"
    )
    
    vocabulary_url: Optional[str] = Field(
        None,
        description="URL of the vocabulary source"
    )
    
    classifications: List[Classification] = Field(
        description="Individual classification results"
    )
    
    total_concepts_considered: int = Field(
        description="Total number of concepts considered"
    )
    
    semantic_filtering_applied: bool = Field(
        description="Whether semantic filtering was applied"
    )
```

### Classification

Individual classification result.

```python
class Classification(BaseModel):
    """Individual classification result."""
    
    uri: str = Field(
        description="URI of the classified concept"
    )
    
    preferred_label: str = Field(
        description="Preferred label of the concept"
    )
    
    alternative_labels: List[str] = Field(
        default_factory=list,
        description="Alternative labels for the concept"
    )
    
    definition: Optional[str] = Field(
        None,
        description="Definition of the concept"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="Explanation for the classification"
    )
    
    broader_concepts: List[str] = Field(
        default_factory=list,
        description="URIs of broader concepts"
    )
    
    narrower_concepts: List[str] = Field(
        default_factory=list,
        description="URIs of narrower concepts"
    )
    
    related_concepts: List[str] = Field(
        default_factory=list,
        description="URIs of related concepts"
    )
```

### DescriptiveFields

Generated descriptive metadata fields.

```python
class DescriptiveFields(BaseModel):
    """Generated descriptive metadata fields."""
    
    title: str = Field(
        description="Generated title for the content"
    )
    
    description: str = Field(
        description="Generated description"
    )
    
    keywords: List[str] = Field(
        description="Extracted keywords"
    )
    
    summary: Optional[str] = Field(
        None,
        description="Generated summary"
    )
    
    learning_objectives: List[str] = Field(
        default_factory=list,
        description="Identified learning objectives"
    )
    
    target_audience: Optional[str] = Field(
        None,
        description="Identified target audience"
    )
    
    difficulty_level: Optional[str] = Field(
        None,
        description="Assessed difficulty level"
    )
    
    estimated_duration: Optional[str] = Field(
        None,
        description="Estimated reading/learning duration"
    )
```

## Resource Suggestion Schemas

### ResourceSuggestionFields

Resource suggestion metadata and recommendations.

```python
class ResourceSuggestionFields(BaseModel):
    """Resource suggestion fields and metadata."""
    
    focus_type: str = Field(
        description="Type of educational focus (content-based, methodical, target-group-based)"
    )
    
    learning_phase: str = Field(
        description="Learning phase (introduction, deepening, practice, assessment)"
    )
    
    filter_suggestions: List[ResourceSuggestionFilterSuggestion] = Field(
        description="Suggested filters for resource discovery"
    )
    
    search_term: str = Field(
        description="Primary search term"
    )
    
    keywords: List[str] = Field(
        description="Additional search keywords"
    )
    
    title: str = Field(
        description="Suggested resource title"
    )
    
    description: str = Field(
        description="Resource description"
    )
```

### ResourceSuggestionFilterSuggestion

Individual filter suggestion for resource discovery.

```python
class ResourceSuggestionFilterSuggestion(BaseModel):
    """Individual filter suggestion."""
    
    vocabulary_url: str = Field(
        description="URL of the vocabulary"
    )
    
    preferred_label: str = Field(
        description="Preferred label of the concept"
    )
    
    alternative_labels: List[str] = Field(
        default_factory=list,
        description="Alternative labels"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the suggestion"
    )
    
    reasoning: str = Field(
        description="Explanation for the suggestion"
    )
```

## Scoring Schemas

### ScoringRequest

Request schema for text quality scoring.

```python
class ScoringRequest(BaseModel):
    """Request model for text scoring."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text content to evaluate"
    )
    
    predefined_metrics: Optional[List[str]] = Field(
        None,
        description="List of predefined metrics to use"
    )
    
    custom_metrics: Optional[List[CustomMetric]] = Field(
        None,
        description="Custom evaluation metrics"
    )
    
    include_improvements: bool = Field(
        False,
        description="Whether to include improvement suggestions"
    )
    
    language: str = Field(
        "de",
        description="Language for evaluation (ISO 639-1 code)"
    )
```

**Example Request:**
```json
{
    "text": "Photosynthesis is the process by which plants convert sunlight into energy.",
    "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],
    "include_improvements": true,
    "language": "de"
}
```

### ScoringResponse

Response schema for scoring operations.

```python
class ScoringResponse(BaseModel):
    """Response model for text scoring."""
    
    scoring_id: str = Field(
        description="Unique identifier for this scoring"
    )
    
    status: Literal["completed", "error"] = Field(
        description="Processing status"
    )
    
    results: Optional[List[MetricEvaluation]] = Field(
        None,
        description="Evaluation results for each metric"
    )
    
    metadata: ScoringMetadata = Field(
        description="Scoring metadata"
    )
    
    processing_time: float = Field(
        description="Processing time in seconds"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if status is 'error'"
    )
```

### CustomMetric

Custom evaluation metric definition.

```python
class CustomMetric(BaseModel):
    """Custom evaluation metric definition."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the metric"
    )
    
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Description of what the metric evaluates"
    )
    
    criteria: List[EvaluationCriterion] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="Evaluation criteria for this metric"
    )
    
    scale_type: Literal["likert_5", "likert_7", "percentage", "binary"] = Field(
        "likert_5",
        description="Type of evaluation scale"
    )
    
    language: str = Field(
        "de",
        description="Language for evaluation prompts"
    )
```

### EvaluationCriterion

Individual evaluation criterion within a metric.

```python
class EvaluationCriterion(BaseModel):
    """Individual evaluation criterion."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the criterion"
    )
    
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Detailed description of the criterion"
    )
    
    weight: float = Field(
        1.0,
        ge=0.1,
        le=10.0,
        description="Weight of this criterion in the overall score"
    )
    
    examples: List[str] = Field(
        default_factory=list,
        description="Examples to guide evaluation"
    )
```

### MetricEvaluation

Evaluation result for a specific metric.

```python
class MetricEvaluation(BaseModel):
    """Evaluation result for a specific metric."""
    
    metric_name: str = Field(
        description="Name of the evaluated metric"
    )
    
    overall_score: float = Field(
        description="Overall score for this metric"
    )
    
    max_possible_score: float = Field(
        description="Maximum possible score"
    )
    
    normalized_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized score (0.0 to 1.0)"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the evaluation"
    )
    
    criterion_scores: List[CriterionScore] = Field(
        description="Scores for individual criteria"
    )
    
    overall_reasoning: str = Field(
        description="Overall reasoning for the score"
    )
    
    suggested_improvements: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )
    
    evaluation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata"
    )
```

### CriterionScore

Score for an individual evaluation criterion.

```python
class CriterionScore(BaseModel):
    """Score for an individual criterion."""
    
    criterion_name: str = Field(
        description="Name of the criterion"
    )
    
    score: float = Field(
        description="Score for this criterion"
    )
    
    max_score: float = Field(
        description="Maximum possible score for this criterion"
    )
    
    weight: float = Field(
        description="Weight of this criterion"
    )
    
    reasoning: str = Field(
        description="Reasoning for this score"
    )
    
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the score"
    )
```

### ScoringMetadata

Metadata about the scoring process.

```python
class ScoringMetadata(BaseModel):
    """Metadata about the scoring process."""
    
    metrics_processed: List[str] = Field(
        description="List of metrics that were processed"
    )
    
    text_length: int = Field(
        description="Length of the input text in characters"
    )
    
    word_count: int = Field(
        description="Number of words in the input text"
    )
    
    language_detected: Optional[str] = Field(
        None,
        description="Detected language of the text"
    )
    
    model_used: str = Field(
        description="AI model used for evaluation"
    )
    
    evaluation_timestamp: str = Field(
        description="ISO timestamp of evaluation"
    )
    
    total_criteria_evaluated: int = Field(
        description="Total number of criteria evaluated"
    )
```

## Health Check Schemas

### HealthResponse

Health check response schema.

```python
class HealthResponse(BaseModel):
    """Health check response."""
    
    status: Literal["healthy", "unhealthy"] = Field(
        description="Overall health status"
    )
    
    timestamp: str = Field(
        description="ISO timestamp of health check"
    )
    
    version: str = Field(
        description="API version"
    )
    
    uptime: float = Field(
        description="Uptime in seconds"
    )
    
    dependencies: Dict[str, DependencyStatus] = Field(
        default_factory=dict,
        description="Status of external dependencies"
    )
```

### DependencyStatus

Status of an external dependency.

```python
class DependencyStatus(BaseModel):
    """Status of an external dependency."""
    
    status: Literal["healthy", "unhealthy", "unknown"] = Field(
        description="Dependency status"
    )
    
    response_time: Optional[float] = Field(
        None,
        description="Response time in seconds"
    )
    
    last_checked: str = Field(
        description="ISO timestamp of last check"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if unhealthy"
    )
```

## Error Schemas

### ErrorResponse

Standard error response schema.

```python
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(
        description="Error type or code"
    )
    
    message: str = Field(
        description="Human-readable error message"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    
    timestamp: str = Field(
        description="ISO timestamp of error"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier for tracking"
    )
    
    validation_errors: Optional[List[ValidationError]] = Field(
        None,
        description="Validation error details"
    )
```

### ValidationError

Detailed validation error information.

```python
class ValidationError(BaseModel):
    """Validation error details."""
    
    field: str = Field(
        description="Field that failed validation"
    )
    
    message: str = Field(
        description="Validation error message"
    )
    
    invalid_value: Any = Field(
        description="The invalid value that was provided"
    )
    
    constraint: Optional[str] = Field(
        None,
        description="The constraint that was violated"
    )
```

## Utility Schemas

### PaginationParams

Pagination parameters for list endpoints.

```python
class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(
        1,
        ge=1,
        description="Page number (1-based)"
    )
    
    size: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of items per page"
    )
    
    sort_by: Optional[str] = Field(
        None,
        description="Field to sort by"
    )
    
    sort_order: Literal["asc", "desc"] = Field(
        "asc",
        description="Sort order"
    )
```

### PaginatedResponse

Paginated response wrapper.

```python
class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    
    items: List[T] = Field(
        description="List of items for current page"
    )
    
    total: int = Field(
        description="Total number of items"
    )
    
    page: int = Field(
        description="Current page number"
    )
    
    size: int = Field(
        description="Items per page"
    )
    
    pages: int = Field(
        description="Total number of pages"
    )
    
    has_next: bool = Field(
        description="Whether there is a next page"
    )
    
    has_prev: bool = Field(
        description="Whether there is a previous page"
    )
```

## Schema Validation Examples

### Request Validation

```python
# Valid classification request
valid_request = {
    "text": "Machine learning in education",
    "mode": "skos",
    "vocabulary_sources": ["discipline"],
    "confidence_threshold": 0.3
}

# This will pass validation
request = ClassificationRequest(**valid_request)

# Invalid request (missing required field)
invalid_request = {
    "mode": "skos"  # Missing 'text' field
}

# This will raise ValidationError
try:
    request = ClassificationRequest(**invalid_request)
except ValidationError as e:
    print(e.errors())
```

### Response Serialization

```python
# Create response object
response = ClassificationResponse(
    classification_id="uuid-123",
    status="completed",
    results=[...],
    processing_time=2.5
)

# Serialize to JSON
json_response = response.model_dump()

# Serialize with exclusions
json_response = response.model_dump(exclude={"processing_time"})

# Serialize with aliases
json_response = response.model_dump(by_alias=True)
```

## Schema Evolution

### Versioning Strategy

The API uses semantic versioning for schema changes:

- **Major version** (v2.0.0): Breaking changes to existing schemas
- **Minor version** (v1.1.0): New optional fields or new schemas
- **Patch version** (v1.0.1): Bug fixes, documentation updates

### Backward Compatibility

- New optional fields can be added without version bump
- Existing fields cannot be removed or changed without major version bump
- Default values ensure backward compatibility
- Deprecated fields are marked and removed in next major version

### Migration Guide

When schemas change, migration guides are provided:

```python
# v1.0 to v1.1 migration example
# New optional field 'resource_suggestion' added

# Old request (still works)
old_request = {
    "text": "Sample text",
    "mode": "skos"
}

# New request (with new field)
new_request = {
    "text": "Sample text", 
    "mode": "skos",
    "resource_suggestion": True  # New optional field
}
```

This comprehensive schema reference provides complete documentation for all data structures used in the Metadata Classification API.
