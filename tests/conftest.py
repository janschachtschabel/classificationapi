"""Test configuration and fixtures."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.services.classification import ClassificationService


@pytest.fixture
def app():
    """Create FastAPI test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_classification_service():
    """Create mock classification service."""
    service = MagicMock(spec=ClassificationService)
    service.classify_text = AsyncMock()
    service.close = AsyncMock()
    return service


@pytest.fixture
def sample_skos_vocabulary():
    """Sample SKOS vocabulary data for testing."""
    return {
        "title": {"de": "Fachbereich", "en": "Subject"},
        "hasTopConcept": [
            {
                "id": "http://example.com/math",
                "prefLabel": {"de": "Mathematik", "en": "Mathematics"},
                "narrower": [{"id": "http://example.com/algebra", "prefLabel": {"de": "Algebra", "en": "Algebra"}}],
            },
            {"id": "http://example.com/science", "prefLabel": {"de": "Naturwissenschaften", "en": "Science"}},
        ],
    }


@pytest.fixture
def sample_classification_request():
    """Sample classification request for testing."""
    return {
        "text": "This is a mathematical text about algebra and equations.",
        "mode": "skos",
        "vocabulary_sources": ["http://example.com/vocab.json"],
        "model": "gpt-4.1-mini",
        "temperature": 0.2,
        "max_tokens": 1000,
    }


@pytest.fixture
def sample_custom_request():
    """Sample custom classification request for testing."""
    return {
        "text": "This is a text about programming and software development.",
        "mode": "custom",
        "custom_categories": {
            "Subject": ["Programming", "Mathematics", "Science"],
            "Level": ["Beginner", "Intermediate", "Advanced"],
        },
    }
