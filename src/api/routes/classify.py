"""Classification endpoint."""

from fastapi import APIRouter, Depends
from loguru import logger

from ...schemas.classification import ClassificationRequest, ClassificationResponse
from ...services.classification import ClassificationService
from ..dependencies import get_classification_service

router = APIRouter(tags=["classification"])


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify text, generate descriptive fields, and suggest educational resources",
    description="""    **Unified endpoint for text classification, metadata generation, and educational resource suggestions.**

    ## Available Processing Modes

    ### 1. SKOS Classification (`mode: "skos"`)
    Classify text against SKOS (Simple Knowledge Organization System) vocabularies.
    - Requires `vocabulary_sources`: List of SKOS vocabulary URLs
    - Returns structured classifications with confidence scores

    ### 2. Custom Classification (`mode: "custom"`)
    Classify text against custom-defined categories.
    - Requires `custom_categories`: Dictionary of category names and their possible values
    - Flexible classification for domain-specific needs

    ### 3. Descriptive Fields Generation (`generate_descriptive_fields: true`)
    Generate structured metadata fields from text content:
    - **Title**: Concise main title (max 50 characters)
    - **Short Title**: Abbreviated title (max 25 characters)
    - **Keywords**: List of relevant keywords
    - **Description**: Comprehensive summary (200-500 characters)

    ### 4. Resource Suggestion (`resource_suggestion: true`)
    Generate intelligent suggestions for educational content databases:
    - **Focus Analysis**: Determines if content is methodical or content-based
    - **Learning Phase**: Identifies learning stage (introduction, deepening, practice, assessment)
    - **Filter Suggestions**: SKOS-based database filters with confidence scores
    - **Search Optimization**: Optimized search terms and keywords
    - **Resource Metadata**: Generated title and description for resource requirements

    ## Usage Examples

    ### SKOS Classification
    ```json
    {
      "text": "Solar panels convert sunlight into electricity using photovoltaic cells.",
      "mode": "skos",
      "vocabulary_sources": ["https://example.com/energy-vocab.rdf"]
    }
    ```

    ### Custom Classification
    ```json
    {
      "text": "This renewable energy project focuses on wind power generation.",
      "mode": "custom",
      "custom_categories": {
        "energy_type": ["renewable", "fossil", "nuclear"],
        "technology": ["solar", "wind", "hydro", "geothermal"]
      }
    }
    ```

    ### Descriptive Fields Only
    ```json
    {
      "text": "Machine learning algorithms analyze data to make predictions.",
      "generate_descriptive_fields": true
    }
    ```

    ### Resource Suggestion Only
    ```json
    {
      "text": "Ich suche Materialien für den Biologieunterricht zum Thema Photosynthese für Klasse 7.",
      "resource_suggestion": true
    }
    ```

    ### Combined Approach (Classification + Descriptive Fields + Resource Suggestion)
    ```json
    {
      "text": "Klimawandel verstehen und Lösungen entwickeln für die Oberstufe.",
      "mode": "skos",
      "vocabulary_sources": ["https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"],
      "generate_descriptive_fields": true,
      "resource_suggestion": true
    }
    ```

    ## Model Configuration
    - **model**: OpenAI model to use (default: gpt-4.1-mini)
    - **temperature**: Creativity level 0.0-2.0 (default: 0.2)
    - **max_tokens**: Maximum response length (default: 15000)

    ## Response Structure
    Returns classification results, optional descriptive fields, and processing metadata.
    """,
    responses={
        200: {
            "description": "Successful classification",
            "content": {
                "application/json": {
                    "example": {
                        "classification_id": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "completed",
                        "results": [
                            {
                                "property": "energy_type",
                                "matches": [
                                    {
                                        "id": "renewable",
                                        "label": "Renewable Energy",
                                        "confidence": 0.95,
                                        "explanation": "Text clearly discusses solar energy, a renewable source",
                                    }
                                ],
                            }
                        ],
                        "descriptive_fields": {
                            "title": "Solar Energy Technology Overview",
                            "short_title": "Solar Energy",
                            "keywords": ["solar", "photovoltaic", "renewable", "electricity"],
                            "description": "Overview of solar panel technology that converts sunlight into electricity using photovoltaic cells, representing a key renewable energy solution for sustainable power generation.",
                        },
                        "resource_suggestion_fields": {
                            "focus_type": "content-based",
                            "learning_phase": "introduction",
                            "filter_suggestions": [
                                {
                                    "vocabulary_url": "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json",
                                    "labels": ["Physics", "Environmental Science"],
                                    "confidence": 0.9,
                                    "reasoning": "Solar energy relates to physics and environmental science topics",
                                }
                            ],
                            "search_term": "solar energy photovoltaic",
                            "keywords": ["solar", "photovoltaic", "renewable energy", "electricity generation"],
                            "title": "Solar Energy Learning Materials",
                            "description": "Educational resources about solar panel technology and photovoltaic energy conversion for renewable energy education.",
                        },
                        "metadata": {
                            "model": "gpt-4.1-mini",
                            "model_settings": {"temperature": 0.2, "max_tokens": 15000},
                            "timestamp": "2024-01-15T10:30:00Z",
                            "processing_time_ms": 1250,
                        },
                    }
                }
            },
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "value_error",
                                "loc": ["body"],
                                "msg": "Either specify a classification mode or enable generate_descriptive_fields",
                                "input": {},
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"error": {"message": "OpenAI API call failed", "type": "OpenAIError"}}}
            },
        },
    },
)
async def classify_text(
    request: ClassificationRequest, service: ClassificationService = Depends(get_classification_service)
) -> ClassificationResponse:
    """
    Classify text based on provided vocabularies or custom categories.

    Args:
        request: Classification request containing text and configuration
        service: Classification service dependency

    Returns:
        ClassificationResponse: Classification results with metadata

    Raises:
        ValidationError: If request validation fails
        VocabularyFetchError: If vocabulary fetching fails
        OpenAIError: If OpenAI API call fails
    """
    logger.info(
        "Classification request received",
        extra={
            "mode": request.mode,
            "text_length": len(request.text),
            "model": request.model,
        },
    )

    try:
        return await service.classify_text(request)
    except Exception as e:
        logger.error(
            f"Unexpected error in classify endpoint: {str(e)}",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "mode": request.mode,
                "text_length": len(request.text),
            },
        )
        raise
