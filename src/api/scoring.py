"""
API endpoints for text scoring and evaluation.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas.scoring import ScoringRequest, ScoringResponse
from ..services.scoring_service import ScoringService
from .dependencies import get_scoring_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["scoring"])


@router.post(
    "/evaluate",
    response_model=ScoringResponse,
    summary="Evaluate text quality using predefined or custom metrics",
    description="""
    Evaluate text quality using predefined or custom scoring metrics.

    **Key Features:**
    - **Universal Prompt System**: All metrics use a sophisticated universal prompt template - no need to specify custom prompts
    - **Detailed Scoring**: Each criterion receives individual scores with explanations
    - **Confidence Metrics**: AI confidence levels for reliability assessment
    - **Improvements Included**: Actionable suggestions for text enhancement
    - **Multi-Metric Support**: Combine predefined and custom metrics in one request
    - **Flexible Validation**: Custom metrics work without requiring explicit flags

    **Supported Predefined Metrics:**
    - `sachrichtigkeit`: Factual accuracy and correctness (binary scale)
    - `neutralitaet`: Neutrality and objectivity assessment (4-point Likert scale)

    **Custom Metrics:**
    - Define your own evaluation criteria with any scale type
    - Universal prompt automatically adapts to your criteria and scale
    - No prompt engineering required - just specify criteria and scale

    **Scale Types:**
    - `binary`: 0-1 scale
    - `likert_3`: 1-3 scale
    - `likert_4`: 1-4 scale
    - `likert_5`: 1-5 scale
    - `percentage`: 0-100 scale
    - `custom`: Define your own min/max values

    **Examples:**

    *Predefined only:*
    ```json
    {"text": "...", "predefined_metrics": ["sachrichtigkeit"]}
    ```

    *Binary custom:*
    ```json
    {
        "text": "...",
        "predefined_metrics": ["custom"],
        "custom_metrics": [{
            "name": "Accessibility",
            "scale": {"min": 0, "max": 1, "type": "binary"},
            "criteria": [{
                "name": "Simple Language",
                "description": "0: Complex jargon, 1: Clear language",
                "weight": 1.0
            }]
        }]
    }
    ```

    *Likert-5 with labels:*
    ```json
    {
        "text": "...",
        "predefined_metrics": ["custom"],
        "custom_metrics": [{
            "name": "Readability",
            "scale": {
                "min": 1, "max": 5, "type": "likert_5",
                "labels": {"1": "Poor", "3": "Average", "5": "Excellent"}
            },
            "criteria": [{
                "name": "Sentence Length",
                "description": "1: Too long, 5: Optimal length",
                "weight": 1.0
            }]
        }]
    }
    ```

    *Custom scale (0-10):*
    ```json
    {
        "text": "...",
        "predefined_metrics": ["custom"],
        "custom_metrics": [{
            "name": "Technical Quality",
            "scale": {"min": 0, "max": 10, "type": "custom"},
            "criteria": [{
                "name": "Code Examples",
                "description": "0: No code, 10: Perfect examples",
                "weight": 2.0
            }]
        }]
    }
    ```

    *Combined with improvements:*
    ```json
    {
        "text": "...",
        "predefined_metrics": ["sachrichtigkeit", "custom"],
        "custom_metrics": [...],
        "include_improvements": true
    }
    ```

    **Adding New YAML Metrics:**
    Create `evaluation_criteria/your_metric.yaml`:
    ```yaml
    evaluation_criteria:
      name: "Your Metric"
      scale: {min: 1, max: 4, type: "likert_4"}
      criteria:
        - name: "Criterion 1"
          description: "1: Poor, 4: Excellent"
          weight: 1.0
    ```
    """,
    responses={
        200: {
            "description": "Text evaluation completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Dies ist ein Beispieltext für die Bewertung.",
                        "results": [
                            {
                                "metric_name": "Sachrichtigkeit",
                                "overall_score": 4.5,
                                "max_possible_score": 6.0,
                                "normalized_score": 0.75,
                                "confidence": 0.85,
                                "criterion_scores": [
                                    {
                                        "name": "Sachliche Richtigkeit",
                                        "score": 1,
                                        "max_score": 1,
                                        "reasoning": "Der Text enthält keine erkennbaren sachlichen Fehler.",
                                        "weight": 1.0,
                                    }
                                ],
                                "overall_reasoning": "Der Text erfüllt die meisten Kriterien für sachliche Richtigkeit.",
                                "suggested_improvements": "Verbesserungsvorschläge...",
                            }
                        ],
                        "processing_time": 2.34,
                        "language": "de",
                        "metadata": {
                            "total_metrics": 1,
                            "predefined_metrics_used": ["sachrichtigkeit"],
                            "custom_metrics_used": [],
                            "openai_calls": 1,
                        },
                    }
                }
            },
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {"detail": "At least one predefined or custom metric must be specified"}
                }
            },
        },
        500: {"description": "Internal server error during evaluation"},
    },
)
async def evaluate_text(
    request: ScoringRequest, scoring_service: ScoringService = Depends(get_scoring_service)
) -> ScoringResponse:
    """
    Evaluate text quality using specified metrics.

    This endpoint provides comprehensive text evaluation using either predefined
    metrics (sachrichtigkeit, neutralitaet) or custom-defined evaluation criteria.
    Each evaluation includes detailed scoring, reasoning, and confidence measures.
    """
    try:
        logger.info(
            f"DEBUG API: Received request - predefined_metrics: {request.predefined_metrics}, custom_metrics: {request.custom_metrics}"
        )
        logger.info(
            f"Evaluating text with {len(request.predefined_metrics or [])} predefined and {len(request.custom_metrics or [])} custom metrics"
        )

        result = await scoring_service.score_text(request)

        logger.info(f"DEBUG API: Service returned {len(result.results)} results")
        logger.info(f"Text evaluation completed in {result.processing_time:.2f}s with {len(result.results)} results")
        return result

    except ValueError as e:
        logger.warning(f"Validation error in text evaluation: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error in text evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during text evaluation"
        )


@router.get(
    "/metrics",
    response_model=dict[str, str],
    summary="Get available predefined metrics",
    description="""
    Retrieve a list of all available predefined evaluation metrics.

    Returns a dictionary mapping metric names to their descriptions.
    These metrics can be used in the `/evaluate` endpoint.

    **Available Metrics:**
    - `sachrichtigkeit`: Evaluates factual accuracy and correctness
    - `neutralitaet`: Evaluates neutrality and objectivity

    **Response Format:**
    ```json
    {
        "sachrichtigkeit": "Bewertung der sachlichen Korrektheit und Genauigkeit von Texten",
        "neutralitaet": "Bewertung der neutralen und objektiven Darstellung von Inhalten"
    }
    ```
    """,
    responses={
        200: {
            "description": "List of available metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "sachrichtigkeit": "Bewertung der sachlichen Korrektheit und Genauigkeit von Texten",
                        "neutralitaet": "Bewertung der neutralen und objektiven Darstellung von Inhalten",
                    }
                }
            },
        }
    },
)
async def get_available_metrics(scoring_service: ScoringService = Depends(get_scoring_service)) -> dict[str, str]:
    """
    Get list of available predefined evaluation metrics.

    Returns a dictionary of metric names and their descriptions that can be
    used with the evaluate endpoint.
    """
    try:
        metrics = scoring_service.get_available_metrics()
        logger.info(f"Retrieved {len(metrics)} available metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error retrieving available metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving available metrics"
        )
