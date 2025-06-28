"""Tests for classification service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.errors import OpenAIError, VocabularyFetchError
from src.core.config import settings
from src.schemas.classification import ClassificationMode, ClassificationRequest, DescriptiveFields
from src.services.classification import ClassificationService, SKOSProcessingStats


class TestClassificationService:
    """Test classification service functionality."""

    @pytest.fixture
    def service(self):
        """Create classification service instance."""
        with patch("src.services.classification.AsyncOpenAI"):
            return ClassificationService()

    @pytest.mark.asyncio
    async def test_parse_skos_vocabulary(self, service, sample_skos_vocabulary):
        """Test SKOS vocabulary parsing."""
        url = "http://example.com/vocab.json"
        result = service._parse_skos_vocabulary(sample_skos_vocabulary, url)

        assert result["name"] == "vocab"
        assert result["property"] == "Fachbereich"
        assert len(result["values"]) == 3  # math, algebra, science

        # Check specific values
        math_value = next(v for v in result["values"] if v["id"] == "http://example.com/math")
        assert math_value["label"] == "Mathematik"

        algebra_value = next(v for v in result["values"] if v["id"] == "http://example.com/algebra")
        assert algebra_value["label"] == "Algebra"

    def test_prepare_custom_categories(self, service):
        """Test custom categories preparation."""
        categories = {"Subject": ["Math", "Science"], "Level": ["Beginner", "Advanced"]}

        result = service._prepare_custom_categories(categories)

        assert len(result) == 2

        subject_vocab = next(v for v in result if v["property"] == "Subject")
        assert len(subject_vocab["values"]) == 2
        assert subject_vocab["values"][0]["label"] == "Math"
        assert subject_vocab["values"][0]["id"] == "custom_Subject_0"

    @pytest.mark.asyncio
    async def test_fetch_vocabularies_success(self, service, sample_skos_vocabulary):
        """Test successful vocabulary fetching."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_skos_vocabulary
        mock_response.raise_for_status.return_value = None

        service.http_client.get = AsyncMock(return_value=mock_response)

        urls = ["http://example.com/vocab.json"]
        result = await service._fetch_vocabularies(urls)

        assert len(result) == 1
        assert result[0]["property"] == "Fachbereich"
        service.http_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_vocabularies_http_error(self, service):
        """Test vocabulary fetching with HTTP error."""
        service.http_client.get = AsyncMock(side_effect=httpx.HTTPError("Network error"))

        urls = ["http://example.com/vocab.json"]

        with pytest.raises(VocabularyFetchError):
            await service._fetch_vocabularies(urls)

    @pytest.mark.asyncio
    async def test_fetch_vocabularies_caching(self, service, sample_skos_vocabulary):
        """Test vocabulary caching functionality."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_skos_vocabulary
        mock_response.raise_for_status.return_value = None

        service.http_client.get = AsyncMock(return_value=mock_response)

        urls = ["http://example.com/vocab.json"]

        # First call
        result1 = await service._fetch_vocabularies(urls)
        # Second call should use cache
        result2 = await service._fetch_vocabularies(urls)

        assert result1 == result2
        # HTTP client should only be called once due to caching
        service.http_client.get.assert_called_once()

    def test_build_classification_prompt(self, service):
        """Test classification prompt building."""
        text = "Test text about mathematics"
        vocab = {
            "property": "Subject",
            "values": [{"id": "math", "label": "Mathematics"}, {"id": "sci", "label": "Science"}],
        }

        prompt = service._build_single_vocabulary_prompt(text, vocab)

        assert "Test text about mathematics" in prompt
        assert "Subject" in prompt
        assert "Mathematics" in prompt
        assert "Science" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_classify_with_openai_success(self, service):
        """Test successful OpenAI classification."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "matches": [
                    {
                        "id": "math",
                        "value": "Mathematics",
                        "confidence": 85,
                        "explanation": "Text discusses math concepts",
                    }
                ]
            }
        )

        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        text = "Test text"
        vocabularies = [{"property": "Subject", "values": [{"id": "math", "label": "Mathematics"}]}]
        skos_stats = SKOSProcessingStats()

        result = await service._classify_with_openai(
            text=text,
            vocabularies=vocabularies,
            model=settings.openai_default_model,
            temperature=0.2,
            max_tokens=1000,
            skos_stats=skos_stats,
        )

        assert len(result) == 1
        assert result[0].property == "Subject"
        assert len(result[0].matches) == 1
        assert result[0].matches[0].id == "math"
        assert result[0].matches[0].label == "Mathematics"
        assert result[0].matches[0].confidence == 0.85  # Converted from percentage

    @pytest.mark.asyncio
    async def test_classify_with_openai_error(self, service):
        """Test OpenAI classification error handling."""
        service.openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        skos_stats = SKOSProcessingStats()

        with pytest.raises(OpenAIError):
            await service._classify_with_openai(
                text="test",
                vocabularies=[{"property": "Subject", "values": [{"id": "test", "label": "Test"}]}],
                model=settings.openai_default_model,
                temperature=0.2,
                max_tokens=1000,
                skos_stats=skos_stats,
            )

    @pytest.mark.asyncio
    async def test_classify_text_skos_mode(self, service, sample_skos_vocabulary):
        """Test full text classification in SKOS mode."""
        # Mock vocabulary fetching
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = sample_skos_vocabulary
        mock_http_response.raise_for_status.return_value = None
        service.http_client.get = AsyncMock(return_value=mock_http_response)

        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "matches": [
                    {
                        "id": "http://example.com/math",
                        "value": "Mathematik",
                        "confidence": 90,
                        "explanation": "Text contains mathematical content",
                    }
                ]
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        request = ClassificationRequest(
            text="This is about mathematics and algebra",
            mode=ClassificationMode.SKOS,
            vocabulary_sources=["http://example.com/vocab.json"],
        )

        result = await service.classify_text(request)

        assert result.status == "completed"
        assert len(result.results) == 1
        assert result.results[0].property == "Fachbereich"
        assert result.metadata.model == settings.openai_default_model  # Actual configured model
        assert "temperature" in result.metadata.model_settings

    @pytest.mark.asyncio
    async def test_classify_text_custom_mode(self, service):
        """Test full text classification in custom mode."""
        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "matches": [
                    {
                        "id": "custom_Subject_0",
                        "value": "Programming",
                        "confidence": 85,
                        "explanation": "Text discusses programming concepts",
                    }
                ]
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        request = ClassificationRequest(
            text="This is about programming and software development",
            mode=ClassificationMode.CUSTOM,
            custom_categories={"Subject": ["Programming", "Mathematics"]},
        )

        result = await service.classify_text(request)

        assert result.status == "completed"
        assert len(result.results) == 1
        assert result.results[0].property == "Subject"
        assert result.results[0].matches[0].label == "Programming"

    def test_build_descriptive_fields_prompt(self, service):
        """Test descriptive fields prompt building."""
        text = "Artificial intelligence is transforming healthcare through advanced diagnostic tools."

        prompt = service._build_descriptive_fields_prompt(text)

        assert text in prompt
        assert "title" in prompt.lower()
        assert "short_title" in prompt.lower()
        assert "keywords" in prompt.lower()
        assert "description" in prompt.lower()
        assert "50 characters" in prompt
        assert "25 characters" in prompt
        assert "200-500 characters" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_generate_descriptive_fields_success(self, service):
        """Test successful descriptive fields generation."""
        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "title": "AI in Healthcare Innovation",
                "short_title": "AI Healthcare",
                "keywords": ["artificial intelligence", "healthcare", "innovation", "technology"],
                "description": "This text explores how artificial intelligence is revolutionizing healthcare through innovative diagnostic tools and treatment methods. The integration of AI technologies is improving patient outcomes and transforming medical practices across various healthcare sectors.",
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        text = "Artificial intelligence is transforming healthcare through advanced diagnostic tools."
        result = await service._generate_descriptive_fields(text, settings.openai_default_model, 0.2, 1000)

        assert isinstance(result, DescriptiveFields)
        assert result.title == "AI in Healthcare Innovation"
        assert result.short_title == "AI Healthcare"
        assert len(result.keywords) == 4
        assert len(result.description) >= 200
        assert len(result.description) <= 500

    @pytest.mark.asyncio
    async def test_generate_descriptive_fields_length_validation(self, service):
        """Test descriptive fields with length validation and correction."""
        # Mock OpenAI response with fields that exceed limits
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "title": "This is a very long title that definitely exceeds the fifty character limit and should be truncated",
                "short_title": "This short title is too long for the limit",
                "keywords": ["ai", "healthcare"],
                "description": "Short description.",
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        text = "Test text for length validation."
        result = await service._generate_descriptive_fields(text, settings.openai_default_model, 0.2, 1000)

        assert isinstance(result, DescriptiveFields)
        assert len(result.title) <= 50
        assert len(result.short_title) <= 25
        assert len(result.description) >= 200
        assert len(result.description) <= 500

    @pytest.mark.asyncio
    async def test_generate_descriptive_fields_error(self, service):
        """Test descriptive fields generation error handling."""
        service.openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(OpenAIError):
            await service._generate_descriptive_fields("test", settings.openai_default_model, 0.2, 1000)

    @pytest.mark.asyncio
    async def test_classify_text_descriptive_fields_only(self, service):
        """Test text classification with only descriptive fields generation."""
        # Mock OpenAI response for descriptive fields
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "title": "Test Article Title",
                "short_title": "Test Article",
                "keywords": ["test", "article", "content"],
                "description": "This is a comprehensive test article that demonstrates the descriptive fields generation functionality. It contains enough content to meet the minimum character requirements for proper validation and testing purposes.",
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        request = ClassificationRequest(
            text="This is a test article about various topics.", generate_descriptive_fields=True
        )

        result = await service.classify_text(request)

        assert result.status == "completed"
        assert result.results == []
        assert result.descriptive_fields is not None
        assert result.descriptive_fields.title == "Test Article Title"
        assert result.descriptive_fields.short_title == "Test Article"
        assert len(result.descriptive_fields.keywords) == 3
        assert len(result.descriptive_fields.description) >= 200

    @pytest.mark.asyncio
    async def test_classify_text_skos_mode_with_stats(self, service, sample_skos_vocabulary):
        """Test SKOS mode classification includes processing statistics in metadata."""
        # Mock vocabulary fetching
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = sample_skos_vocabulary
        mock_http_response.raise_for_status.return_value = None
        service.http_client.get = AsyncMock(return_value=mock_http_response)

        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "matches": [
                    {
                        "id": "http://example.com/math",
                        "value": "Mathematik",
                        "confidence": 90,
                        "explanation": "Text contains mathematical content",
                    }
                ]
            }
        )
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        request = ClassificationRequest(
            text="This is about mathematics and algebra",
            mode=ClassificationMode.SKOS,
            vocabulary_sources=["http://example.com/vocab.json"],
        )

        result = await service.classify_text(request)

        # Verify basic response structure
        assert result.status == "completed"
        assert len(result.results) == 1
        assert result.results[0].property == "Fachbereich"

        # Verify SKOS stats show semantic filtering behavior
        assert result.metadata.skos_concepts_found is not None
        assert result.metadata.skos_concepts_processed is not None
        assert result.metadata.semantic_filtering_used is not None

        # For this small vocabulary (3 concepts), semantic filtering should not be applied
        assert result.metadata.skos_concepts_found == 3  # Math, Algebra, Science
        assert result.metadata.skos_concepts_processed == 3  # All 3 concepts processed (2 top + 1 narrower)
        assert result.metadata.semantic_filtering_used  # Filtering is enabled by default
        assert result.metadata.semantic_filtering_info is not None  # Filtering info is provided

    @pytest.mark.asyncio
    async def test_classify_text_disable_semantic_filtering(self, service):
        """Test that enable_semantic_filtering parameter works correctly when disabled."""
        # Mock a large vocabulary that would normally trigger block processing
        large_vocab_data = {
            "title": {"en": "Test Vocabulary"},
            "hasTopConcept": [
                {"id": f"concept_{i}", "@type": "skos:Concept", "prefLabel": {"en": f"Concept {i}"}}
                for i in range(250)  # More than large_vocab_threshold (200)
            ],
        }

        # Mock successful OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = json.dumps(
            {"matches": [{"id": "concept_1", "value": "Concept 1", "confidence": 85, "explanation": "Test reasoning"}]}
        )

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_vocab_data
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            # Mock OpenAI client
            service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            # Test with enable_semantic_filtering=False
            request = ClassificationRequest(
                text="Test text about programming",
                mode=ClassificationMode.SKOS,
                vocabulary_sources=["http://example.com/vocab"],
                enable_semantic_filtering=False,
            )

            result = await service.classify_text(request)

            # Verify response structure
            assert result.status == "completed"
            assert len(result.results) == 1
            assert result.results[0].property == "Test Vocabulary"

            # Verify SKOS stats show semantic filtering was disabled
            assert result.metadata.skos_concepts_found == 250
            assert result.metadata.skos_concepts_processed == 250  # All 250 concepts processed (no narrower concepts)
            assert result.metadata.semantic_filtering_used is False
            assert result.metadata.semantic_filtering_disabled is True  # Semantic filtering was disabled
            assert result.metadata.semantic_filtering_info is None
