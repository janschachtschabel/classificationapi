# Testing Guide

Comprehensive testing guide for the Metadata Classification API.

## Test Structure

The project uses pytest for testing with the following structure:

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_api_health.py       # Health endpoint tests
├── test_classification.py   # Classification endpoint tests
├── test_scoring.py          # Scoring endpoint tests
├── test_schemas.py          # Schema validation tests
├── test_services.py         # Service layer tests
└── test_integration.py      # Integration tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_classification.py

# Run specific test function
pytest tests/test_classification.py::test_classify_text_skos_mode

# Run tests matching pattern
pytest -k "classification"
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=src

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate coverage report with missing lines
pytest --cov=src --cov-report=term-missing
```

### Test Configuration

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto

# Run tests with specific markers
pytest -m "unit"
pytest -m "integration"

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from src.schemas.classification import ClassificationRequest

class TestClassificationRequest:
    """Test classification request schema validation."""
    
    def test_valid_skos_request(self):
        """Test valid SKOS classification request."""
        request = ClassificationRequest(
            text="Machine learning in education",
            mode="skos",
            vocabulary_sources=["discipline", "educational_context"]
        )
        
        assert request.text == "Machine learning in education"
        assert request.mode == "skos"
        assert len(request.vocabulary_sources) == 2
    
    def test_valid_custom_request(self):
        """Test valid custom classification request."""
        request = ClassificationRequest(
            text="Educational content",
            mode="custom",
            categories=["Science", "Technology", "Education"]
        )
        
        assert request.mode == "custom"
        assert len(request.categories) == 3
    
    def test_invalid_empty_text(self):
        """Test validation fails for empty text."""
        with pytest.raises(ValidationError) as exc_info:
            ClassificationRequest(text="", mode="skos")
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    def test_invalid_mode_combination(self):
        """Test validation fails for invalid mode combinations."""
        with pytest.raises(ValidationError) as exc_info:
            ClassificationRequest(
                text="Test text",
                mode="skos",
                categories=["Category1"]  # Invalid for SKOS mode
            )
        
        assert "Categories should not be provided for SKOS mode" in str(exc_info.value)
```

### Integration Tests

Test component interactions:

```python
# tests/test_integration.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

class TestAPIIntegration:
    """Test API endpoint integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_full_classification_workflow(self, client):
        """Test complete classification workflow."""
        # Test health check first
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Test classification
        classification_data = {
            "text": "Introduction to machine learning algorithms",
            "mode": "skos",
            "generate_descriptive_fields": True
        }
        
        response = client.post("/classify", json=classification_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "classification_id" in data
        assert "results" in data
        assert "descriptive_fields" in data
        
        # Verify classification results
        assert len(data["results"]) > 0
        for result in data["results"]:
            assert "vocabulary_name" in result
            assert "classifications" in result
    
    def test_scoring_integration(self, client):
        """Test scoring endpoint integration."""
        scoring_data = {
            "text": "Educational content about photosynthesis",
            "predefined_metrics": ["sachrichtigkeit"],
            "include_improvements": True
        }
        
        response = client.post("/scoring/evaluate", json=scoring_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        assert result["metric_name"] == "sachrichtigkeit"
        assert "overall_score" in result
        assert "normalized_score" in result
```

### Service Layer Tests

Test business logic:

```python
# tests/test_services.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.classification import ClassificationService
from src.core.config import Settings

class TestClassificationService:
    """Test classification service functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.openai_api_key = "test-key"
        settings.openai_default_model = "gpt-4.1-mini"
        settings.default_temperature = 0.1
        settings.default_max_tokens = 1000
        settings.openai_timeout_seconds = 30
        return settings
    
    @pytest.fixture
    def service(self, mock_settings):
        """Create classification service with mocked dependencies."""
        with patch('src.services.classification.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            service = ClassificationService(mock_settings)
            service.openai_client = mock_client
            return service
    
    @pytest.mark.asyncio
    async def test_classify_text_success(self, service):
        """Test successful text classification."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "classifications": [
                {
                    "preferred_label": "Machine Learning",
                    "confidence": 0.95,
                    "reasoning": "Text discusses ML algorithms"
                }
            ]
        }
        '''
        
        service.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test classification
        result = await service.classify_text(
            text="Introduction to machine learning",
            mode="custom",
            categories=["Machine Learning", "Data Science"]
        )
        
        assert result["status"] == "completed"
        assert len(result["results"]) > 0
        
        # Verify OpenAI was called correctly
        service.openai_client.chat.completions.create.assert_called_once()
        call_args = service.openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4.1-mini"
        assert call_args[1]["temperature"] == 0.1
    
    @pytest.mark.asyncio
    async def test_classify_text_openai_error(self, service):
        """Test handling of OpenAI API errors."""
        # Mock OpenAI error
        service.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await service.classify_text(
            text="Test text",
            mode="custom",
            categories=["Category1"]
        )
        
        assert result["status"] == "error"
        assert "API Error" in result["error"]
```

## Test Fixtures

### Shared Fixtures

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from src.main import app
from src.core.config import Settings

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_default_model = "gpt-4.1-mini"
    settings.default_temperature = 0.1
    settings.default_max_tokens = 1000
    settings.openai_timeout_seconds = 30
    settings.log_level = "INFO"
    settings.api_port = 8000
    return settings

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Introduction to machine learning algorithms",
        "Photosynthesis in plant biology",
        "European history in the 19th century",
        "Quantum mechanics for beginners"
    ]

@pytest.fixture
def sample_classification_request():
    """Sample classification request data."""
    return {
        "text": "Machine learning in educational technology",
        "mode": "skos",
        "vocabulary_sources": ["discipline", "educational_context"],
        "generate_descriptive_fields": True
    }

@pytest.fixture
def sample_scoring_request():
    """Sample scoring request data."""
    return {
        "text": "Educational content about photosynthesis",
        "predefined_metrics": ["sachrichtigkeit"],
        "include_improvements": True
    }
```

### Mock Data Fixtures

```python
@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": '''
                    {
                        "classifications": [
                            {
                                "preferred_label": "Computer Science",
                                "confidence": 0.92,
                                "reasoning": "Text discusses machine learning"
                            }
                        ]
                    }
                    '''
                }
            }
        ]
    }

@pytest.fixture
def mock_vocabulary_data():
    """Mock SKOS vocabulary data."""
    return {
        "discipline": {
            "concepts": [
                {
                    "uri": "http://example.org/computer-science",
                    "preferred_label": "Computer Science",
                    "alternative_labels": ["CS", "Computing"],
                    "definition": "Study of computational systems"
                }
            ]
        }
    }
```

## Mocking Strategies

### External API Mocking

```python
import pytest
from unittest.mock import patch, Mock, AsyncMock

class TestExternalAPIMocking:
    """Test external API mocking strategies."""
    
    @pytest.mark.asyncio
    @patch('src.services.classification.OpenAI')
    async def test_openai_api_mock(self, mock_openai_class):
        """Test OpenAI API mocking."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test service
        from src.services.classification import ClassificationService
        service = ClassificationService(mock_settings())
        
        result = await service.classify_text("test", "custom", ["cat1"])
        
        # Verify mock was called
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.services.vocabulary_loader.requests.get')
    def test_vocabulary_loading_mock(self, mock_get):
        """Test vocabulary loading with mocked HTTP requests."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"concepts": []}
        mock_get.return_value = mock_response
        
        from src.services.vocabulary_loader import VocabularyLoader
        loader = VocabularyLoader()
        
        result = loader.load_vocabulary("test-url")
        
        assert result is not None
        mock_get.assert_called_once_with("test-url")
```

### Database Mocking

```python
@pytest.fixture
def mock_cache():
    """Mock TTL cache for testing."""
    cache_data = {}
    
    class MockCache:
        def get(self, key, default=None):
            return cache_data.get(key, default)
        
        def __setitem__(self, key, value):
            cache_data[key] = value
        
        def __getitem__(self, key):
            return cache_data[key]
        
        def __contains__(self, key):
            return key in cache_data
    
    return MockCache()
```

## Performance Testing

### Load Testing

```python
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance and load testing."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client):
        """Test handling of concurrent requests."""
        
        async def make_request():
            response = test_client.post("/classify", json={
                "text": "Test text for performance",
                "mode": "skos"
            })
            return response.status_code
        
        # Create multiple concurrent requests
        tasks = [make_request() for _ in range(10)]
        start_time = time.time()
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Verify all requests succeeded
        assert all(status == 200 for status in results)
        
        # Verify reasonable response time
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within 30 seconds
    
    @pytest.mark.slow
    def test_memory_usage(self, test_client):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for i in range(100):
            response = test_client.post("/classify", json={
                "text": f"Test text number {i}",
                "mode": "skos"
            })
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
```

## Test Data Management

### Test Data Files

```python
# tests/data/sample_texts.json
{
    "educational_texts": [
        {
            "text": "Introduction to calculus for engineering students",
            "expected_subjects": ["Mathematics", "Engineering"],
            "expected_level": "undergraduate"
        },
        {
            "text": "Photosynthesis process in plant biology",
            "expected_subjects": ["Biology", "Life Sciences"],
            "expected_level": "secondary"
        }
    ],
    "scoring_texts": [
        {
            "text": "Well-structured educational content with clear examples",
            "expected_scores": {
                "sachrichtigkeit": 0.8,
                "neutralitaet": 0.9
            }
        }
    ]
}
```

### Data Loading Utilities

```python
# tests/utils/data_loader.py
import json
import os
from pathlib import Path

def load_test_data(filename: str):
    """Load test data from JSON file."""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / filename
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_sample_texts(category: str = "educational_texts"):
    """Get sample texts for testing."""
    data = load_test_data("sample_texts.json")
    return data.get(category, [])
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        
      - id: mypy
        name: mypy
        entry: mypy src
        language: system
        pass_filenames: false
        always_run: true
```

## Test Best Practices

### Writing Effective Tests

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests clearly with setup, execution, and verification
3. **Single Responsibility**: Each test should verify one specific behavior
4. **Test Independence**: Tests should not depend on each other
5. **Mock External Dependencies**: Isolate the code under test

### Coverage Guidelines

- **Aim for 80%+ overall coverage**
- **100% coverage for critical paths**
- **Focus on business logic coverage**
- **Don't ignore edge cases**

### Test Maintenance

- **Keep tests simple and readable**
- **Update tests when requirements change**
- **Remove obsolete tests**
- **Refactor test code like production code**

This comprehensive testing guide ensures robust validation of all API components and maintains high code quality standards.
