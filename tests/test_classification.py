"""Tests for classification endpoint."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.schemas.classification import (
    ClassificationMatch,
    ClassificationMetadata,
    ClassificationResponse,
    ClassificationResult,
    DescriptiveFields,
    ResourceSuggestionFields,
    ResourceSuggestionFilterSuggestion,
)


class TestClassificationEndpoint:
    """Test classification endpoint functionality."""

    def test_classify_skos_mode_success(self, client: TestClient, sample_classification_request):
        """Test successful SKOS classification."""
        # Create mock response
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440000",
            status="completed",
            results=[
                ClassificationResult(
                    property="Subject",
                    matches=[
                        ClassificationMatch(
                            id="http://example.com/math",
                            label="Mathematics",
                            confidence=0.95,
                            explanation="Text discusses mathematical concepts",
                        )
                    ],
                )
            ],
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=1500,
            ),
        )

        # Mock the classification service
        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            response = client.post("/classify", json=sample_classification_request)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert len(data["results"]) == 1
            assert data["results"][0]["property"] == "Subject"
            assert len(data["results"][0]["matches"]) == 1
            assert data["metadata"]["model"] == "gpt-4.1-mini"

    def test_classify_custom_mode_success(self, client: TestClient, sample_custom_request):
        """Test successful custom classification."""
        # Create mock response
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440001",
            status="completed",
            results=[
                ClassificationResult(
                    property="Subject",
                    matches=[
                        ClassificationMatch(
                            id="custom_Subject_0",
                            label="Programming",
                            confidence=0.88,
                            explanation="Text discusses programming concepts",
                        )
                    ],
                )
            ],
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=1200,
            ),
        )

        # Mock the classification service
        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            response = client.post("/classify", json=sample_custom_request)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert len(data["results"]) == 1
            assert data["results"][0]["property"] == "Subject"
            assert data["results"][0]["matches"][0]["label"] == "Programming"

    def test_classify_missing_vocabulary_sources(self, client: TestClient):
        """Test validation error when vocabulary_sources missing for SKOS mode without descriptive_fields."""
        request_data = {
            "text": "Test text",
            "mode": "skos",
            # Missing vocabulary_sources and generate_descriptive_fields=false
        }

        response = client.post("/classify", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "vocabulary_sources is required for 'skos' mode" in str(data["detail"])

    def test_classify_missing_custom_categories(self, client: TestClient):
        """Test validation error when custom_categories missing for custom mode without descriptive_fields."""
        request_data = {
            "text": "Test text",
            "mode": "custom",
            # Missing custom_categories and generate_descriptive_fields=false
        }

        response = client.post("/classify", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "custom_categories is required for 'custom' mode" in str(data["detail"])

    def test_classify_descriptive_fields_only(self, client: TestClient):
        """Test successful descriptive fields generation without classification mode."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440002",
            status="completed",
            results=[],
            descriptive_fields=DescriptiveFields(
                title="Test Article Title",
                short_title="Test Article",
                keywords=["test", "article", "content"],
                description="This is a comprehensive description of the test article content that meets the minimum character requirement for the description field validation. It provides detailed information about the article's purpose and content structure for testing the descriptive fields generation functionality within the classification API system.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=2000,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Test article content for descriptive fields generation.",
                "generate_descriptive_fields": True,
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert data["results"] == []
            assert "descriptive_fields" in data
            assert data["descriptive_fields"]["title"] == "Test Article Title"
            assert data["descriptive_fields"]["short_title"] == "Test Article"
            assert len(data["descriptive_fields"]["keywords"]) == 3
            assert len(data["descriptive_fields"]["description"]) >= 200

    def test_classify_skos_with_empty_vocabulary_sources_and_descriptive_fields(self, client: TestClient):
        """Test SKOS mode with empty vocabulary_sources but descriptive_fields enabled."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440003",
            status="completed",
            results=[],
            descriptive_fields=DescriptiveFields(
                title="SKOS Test Content",
                short_title="SKOS Test",
                keywords=["skos", "test", "classification"],
                description="This test demonstrates the SKOS classification mode with empty vocabulary sources but descriptive fields generation enabled for comprehensive content analysis. It provides detailed information about the content structure and purpose for testing the descriptive fields generation functionality within the classification API system.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=2500,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Test content for SKOS classification with descriptive fields.",
                "mode": "skos",
                "vocabulary_sources": [],
                "generate_descriptive_fields": True,
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert data["results"] == []
            assert "descriptive_fields" in data
            assert data["descriptive_fields"]["title"] == "SKOS Test Content"

    def test_classify_custom_without_categories_but_with_descriptive_fields(self, client: TestClient):
        """Test custom mode without custom_categories but with descriptive_fields enabled."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440004",
            status="completed",
            results=[],
            descriptive_fields=DescriptiveFields(
                title="Custom Mode Test",
                short_title="Custom Test",
                keywords=["custom", "classification", "test"],
                description="This test validates the custom classification mode without predefined categories but with descriptive fields generation enabled for flexible content analysis workflows. It provides detailed information about the content structure and purpose for testing the descriptive fields generation functionality within the classification API system.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=1800,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Test content for custom classification with descriptive fields.",
                "mode": "custom",
                "generate_descriptive_fields": True,
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert data["results"] == []
            assert "descriptive_fields" in data
            assert data["descriptive_fields"]["title"] == "Custom Mode Test"

    def test_classify_combined_classification_and_descriptive_fields(self, client: TestClient):
        """Test combined classification and descriptive fields generation."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440005",
            status="completed",
            results=[
                ClassificationResult(
                    property="Subject",
                    matches=[
                        ClassificationMatch(
                            id="custom_Subject_0",
                            label="Technology",
                            confidence=0.92,
                            explanation="Text discusses technology topics",
                        )
                    ],
                )
            ],
            descriptive_fields=DescriptiveFields(
                title="Technology Innovation Article",
                short_title="Tech Innovation",
                keywords=["technology", "innovation", "digital"],
                description="This article explores various aspects of technology innovation and digital transformation in modern business environments, highlighting key trends and future developments. It provides detailed information about the content structure and purpose for testing the descriptive fields generation functionality within the classification API system.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=3000,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Technology innovation is driving digital transformation across industries.",
                "mode": "custom",
                "custom_categories": {"Subject": ["Technology", "Business", "Innovation"]},
                "generate_descriptive_fields": True,
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert len(data["results"]) == 1
            assert data["results"][0]["matches"][0]["label"] == "Technology"
            assert "descriptive_fields" in data
            assert data["descriptive_fields"]["title"] == "Technology Innovation Article"

    def test_classify_validation_error_no_mode_no_descriptive_fields(self, client: TestClient):
        """Test validation error when neither mode nor descriptive_fields is specified."""
        request_data = {
            "text": "Test text"
            # Missing both mode and generate_descriptive_fields
        }

        response = client.post("/classify", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert (
            "At least one of 'mode', 'generate_descriptive_fields', or 'resource_suggestion' must be specified"
            in str(data["detail"])
        )

    def test_classify_invalid_text_length(self, client: TestClient):
        """Test validation error for text that's too long."""
        request_data = {
            "text": "x" * 60000,  # Exceeds max length
            "mode": "custom",
            "custom_categories": {"Subject": ["Test"]},
        }

        response = client.post("/classify", json=request_data)

        assert response.status_code == 422

    def test_classify_invalid_temperature(self, client: TestClient):
        """Test validation error for invalid temperature."""
        request_data = {
            "text": "Test text",
            "mode": "custom",
            "custom_categories": {"Subject": ["Test"]},
            "temperature": 3.0,  # Exceeds max value
        }

        response = client.post("/classify", json=request_data)

        assert response.status_code == 422

    def test_classify_resource_suggestion_only(self, client: TestClient):
        """Test resource suggestion generation without classification mode."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440006",
            status="completed",
            results=[],
            resource_suggestion_fields=ResourceSuggestionFields(
                focus_type="content-based",
                focus_explanation="The text focuses on specific subject matter content.",
                learning_phase="elaboration",
                phase_explanation="Content is suitable for the elaboration phase of learning.",
                filter_suggestions=[
                    ResourceSuggestionFilterSuggestion(
                        vocabulary_url="http://example.com/vocab",
                        vocabulary_name="Test Vocabulary",
                        values=["test_id_1"],
                        uris=[],
                        labels=["Test Label"],
                        confidence=0.9,
                        reasoning="Test reasoning for filter suggestion",
                    )
                ],
                search_term="test search term",
                keywords=["test", "keyword", "example"],
                title="Test Resource Title",
                description="Test resource description for educational content.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=2000,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Educational content about mathematics and problem solving.",
                "resource_suggestion": True,
                "vocabulary_sources": [],  # Empty list to avoid external API calls
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert "resource_suggestion_fields" in data
            assert data["resource_suggestion_fields"]["focus_type"] == "content-based"
            assert data["resource_suggestion_fields"]["learning_phase"] == "elaboration"
            assert len(data["resource_suggestion_fields"]["filter_suggestions"]) == 1
            assert data["resource_suggestion_fields"]["title"] == "Test Resource Title"

    def test_classify_combined_classification_and_resource_suggestion(self, client: TestClient):
        """Test combined classification and resource suggestion generation."""
        mock_response = ClassificationResponse(
            classification_id="550e8400-e29b-41d4-a716-446655440007",
            status="completed",
            results=[
                ClassificationResult(
                    property="Subject",
                    matches=[
                        ClassificationMatch(
                            id="http://example.com/science",
                            label="Science",
                            confidence=0.92,
                            explanation="Text discusses scientific concepts",
                        )
                    ],
                )
            ],
            resource_suggestion_fields=ResourceSuggestionFields(
                focus_type="methodical",
                focus_explanation="The content focuses on teaching methods and approaches.",
                learning_phase="practice",
                phase_explanation="Content is suitable for practice and application.",
                filter_suggestions=[
                    ResourceSuggestionFilterSuggestion(
                        vocabulary_url="http://example.com/methods",
                        vocabulary_name="Teaching Methods",
                        values=["method_1", "method_2"],
                        uris=[],
                        labels=["Interactive Learning", "Problem-Based Learning"],
                        confidence=0.85,
                        reasoning="Methods align with the educational approach described.",
                    )
                ],
                search_term="interactive science learning",
                keywords=["science", "interactive", "learning", "methods"],
                title="Interactive Science Learning Methods",
                description="Resources for interactive science education and problem-based learning approaches.",
            ),
            metadata=ClassificationMetadata(
                model="gpt-4.1-mini",
                model_settings={"temperature": 0.2, "max_tokens": 1000},
                timestamp="2025-06-24T11:00:00",
                processing_time_ms=3500,
            ),
        )

        with patch(
            "src.services.classification.ClassificationService.classify_text", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = mock_response

            request_data = {
                "text": "Interactive science learning through hands-on experiments and problem-solving activities.",
                "mode": "skos",
                "vocabulary_sources": [],  # Empty list to avoid external API calls
                "resource_suggestion": True,
            }

            response = client.post("/classify", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "completed"
            assert len(data["results"]) == 1
            assert data["results"][0]["matches"][0]["label"] == "Science"
            assert "resource_suggestion_fields" in data
            assert data["resource_suggestion_fields"]["focus_type"] == "methodical"
            assert len(data["resource_suggestion_fields"]["filter_suggestions"]) == 1
            assert "Interactive Learning" in data["resource_suggestion_fields"]["filter_suggestions"][0]["labels"]
