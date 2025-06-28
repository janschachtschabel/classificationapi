# Contributing Guide

Thank you for your interest in contributing to the Classification API! This guide will help you get started with development and contributions.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- OpenAI API key for testing

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd classificationapi
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

5. **Run tests**
   ```bash
   pytest
   ```

6. **Start development server**
   ```bash
   uvicorn src.main:app --reload
   ```

## 📁 Project Structure

```
classificationapi/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   ├── utils/             # Utility functions
│   └── main.py           # FastAPI application
├── tests/                 # Test suite
├── docs/                  # Documentation
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── pyproject.toml        # Project configuration
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_classification_api.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_classification"
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints and service interactions
- **Performance Tests**: Test response times and resource usage

### Writing Tests

Follow these conventions:

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

class TestClassificationAPI:
    """Test classification endpoint functionality."""
    
    def test_skos_classification_success(self):
        """Test successful SKOS classification."""
        response = client.post("/classify", json={
            "text": "Sample educational content",
            "mode": "skos"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "classification_id" in data
        assert "results" in data
        assert len(data["results"]) > 0
    
    def test_invalid_text_length(self):
        """Test validation for text length."""
        long_text = "x" * 10001  # Exceeds limit
        
        response = client.post("/classify", json={
            "text": long_text,
            "mode": "skos"
        })
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_async_classification(self):
        """Test asynchronous classification processing."""
        # Async test implementation
        pass
```

## 🔧 Code Standards

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Check linting
flake8 src tests

# Type checking
mypy src
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

### Type Hints

All code must include comprehensive type hints:

```python
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

def classify_text(
    text: str, 
    categories: Dict[str, List[str]],
    confidence_threshold: float = 0.5
) -> Optional[Dict[str, Union[str, float]]]:
    """Classify text with custom categories.
    
    Args:
        text: Input text to classify
        categories: Dictionary of category groups and options
        confidence_threshold: Minimum confidence for results
        
    Returns:
        Classification results or None if no matches
        
    Raises:
        ValueError: If text is empty or categories invalid
    """
    pass
```

## 📝 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def process_classification_request(request: ClassificationRequest) -> ClassificationResponse:
    """Process a classification request.
    
    This function handles the complete classification workflow including
    text preprocessing, model inference, and result formatting.
    
    Args:
        request: The classification request containing text and parameters
        
    Returns:
        ClassificationResponse: Structured classification results
        
    Raises:
        ValidationError: If request validation fails
        OpenAIError: If OpenAI API call fails
        
    Example:
        >>> request = ClassificationRequest(text="Sample text", mode="skos")
        >>> response = process_classification_request(request)
        >>> print(response.classification_id)
    """
    pass
```

### API Documentation

- All endpoints must have comprehensive OpenAPI documentation
- Include request/response examples
- Document all possible error responses
- Add usage examples in docstrings

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment details**
   - Python version
   - Operating system
   - Package versions

2. **Reproduction steps**
   - Minimal code example
   - Input data that causes the issue
   - Expected vs actual behavior

3. **Error information**
   - Full error traceback
   - Log messages
   - API response details

### Feature Requests

For new features, provide:

1. **Use case description**
2. **Proposed API design**
3. **Implementation considerations**
4. **Backward compatibility impact**

## 🔄 Pull Request Process

### Before Submitting

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes with tests**
   - Add comprehensive tests for new functionality
   - Update existing tests if needed
   - Ensure all tests pass

3. **Update documentation**
   - Add docstrings to new functions
   - Update API documentation
   - Add examples if applicable

4. **Check code quality**
   ```bash
   black src tests
   isort src tests
   flake8 src tests
   mypy src
   pytest --cov=src
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing completed

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] README updated if needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No new linting errors
- [ ] Type hints added for new code
```

## 🏗️ Architecture Guidelines

### Service Layer Pattern

Organize business logic in service classes:

```python
class ClassificationService:
    """Service for handling text classification operations."""
    
    def __init__(self, openai_client: OpenAI, cache: TTLCache):
        self.openai_client = openai_client
        self.cache = cache
    
    async def classify_with_skos(
        self, 
        text: str, 
        vocabulary_sources: List[str]
    ) -> ClassificationResult:
        """Classify text using SKOS vocabularies."""
        pass
```

### Error Handling

Use structured error handling:

```python
from src.exceptions import ClassificationError, ValidationError

class ClassificationService:
    async def classify_text(self, request: ClassificationRequest) -> ClassificationResponse:
        try:
            # Classification logic
            pass
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ClassificationError("Classification service unavailable") from e
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ClassificationError("Internal classification error") from e
```

### Async/Await Patterns

Use async/await for I/O operations:

```python
async def fetch_vocabulary(url: str) -> Dict[str, Any]:
    """Fetch SKOS vocabulary from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

async def classify_with_multiple_vocabularies(
    text: str, 
    vocabulary_urls: List[str]
) -> List[ClassificationResult]:
    """Classify text with multiple vocabularies concurrently."""
    tasks = [
        classify_with_vocabulary(text, url) 
        for url in vocabulary_urls
    ]
    return await asyncio.gather(*tasks)
```

## 🚀 Performance Guidelines

### Caching Strategy

Implement appropriate caching:

```python
from cachetools import TTLCache
import hashlib

class VocabularyCache:
    def __init__(self, maxsize: int = 100, ttl: int = 3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get_vocabulary(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached vocabulary or None if not found."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        return self.cache.get(cache_key)
    
    def set_vocabulary(self, url: str, vocabulary: Dict[str, Any]) -> None:
        """Cache vocabulary data."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        self.cache[cache_key] = vocabulary
```

### Resource Management

Use context managers for resource cleanup:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def openai_client_context():
    """Context manager for OpenAI client."""
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        yield client
    finally:
        await client.close()
```

## 🔒 Security Guidelines

### API Key Management

- Never commit API keys to version control
- Use environment variables for secrets
- Implement key rotation procedures
- Add API key validation

### Input Validation

Validate all inputs thoroughly:

```python
from pydantic import BaseModel, validator

class ClassificationRequest(BaseModel):
    text: str
    mode: str
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long (max 10,000 characters)')
        return v.strip()
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['skos', 'custom']:
            raise ValueError('Mode must be "skos" or "custom"')
        return v
```

## 📊 Monitoring and Logging

### Structured Logging

Use structured logging with context:

```python
import structlog

logger = structlog.get_logger()

async def classify_text(request: ClassificationRequest) -> ClassificationResponse:
    """Classify text with comprehensive logging."""
    
    logger.info(
        "Classification request received",
        text_length=len(request.text),
        mode=request.mode,
        request_id=request.id
    )
    
    try:
        result = await process_classification(request)
        
        logger.info(
            "Classification completed successfully",
            request_id=request.id,
            result_count=len(result.classifications),
            processing_time=result.processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Classification failed",
            request_id=request.id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

## 🤝 Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Documentation**: Comprehensive guides and examples

### Code of Conduct

We follow a code of conduct that ensures a welcoming environment for all contributors. Please be respectful, inclusive, and constructive in all interactions.

Thank you for contributing to the Classification API! 🎉
