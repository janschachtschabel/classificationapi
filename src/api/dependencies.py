"""API dependencies."""

from functools import lru_cache

from ..core.config import settings
from ..services.classification import ClassificationService
from ..services.scoring_service import ScoringService


@lru_cache
def get_classification_service() -> ClassificationService:
    """
    Get classification service instance.

    Returns:
        ClassificationService: Singleton service instance
    """
    return ClassificationService()


@lru_cache
def get_scoring_service() -> ScoringService:
    """
    Get scoring service instance.

    Returns:
        ScoringService: Singleton service instance
    """
    return ScoringService(settings)
