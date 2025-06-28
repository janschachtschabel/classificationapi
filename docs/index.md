# Classification API

Welcome to the Classification API documentation! This API provides zero-shot classification of educational metadata using OpenAI's language models, with integrated resource suggestion capabilities.

## Overview

The Classification API allows you to:

- **Classify text** using SKOS vocabularies or custom categories
- **Generate descriptive metadata** (titles, descriptions, keywords)
- **Get resource suggestions** with intelligent filtering and search recommendations

### Classification Modes

- **SKOS Vocabularies**: Structured vocabularies from SKOHub or other SKOS-compliant sources
- **Custom Categories**: Your own predefined category lists

## Key Features

- ✅ **Zero-shot Classification**: No training required, works with any text
- ✅ **SKOS Support**: Direct integration with SKOHub vocabularies
- ✅ **Custom Categories**: Use your own classification schemes
- ✅ **Resource Suggestions**: AI-powered resource filtering and search recommendations
- ✅ **Descriptive Metadata**: Auto-generate titles, descriptions, and keywords
- ✅ **Unified Endpoint**: Single `/classify` endpoint for all functionality
- ✅ **Multiple Models**: Support for various OpenAI models
- ✅ **Caching**: Vocabulary caching for improved performance
- ✅ **Comprehensive Logging**: Detailed logging and error handling
- ✅ **Type Safety**: Full type hints and Pydantic validation

## Quick Examples

### Basic Classification

```python
import httpx

# Classification request
request_data = {
    "text": "This course covers linear algebra, calculus, and statistics.",
    "mode": "skos",
    "vocabulary_sources": [
        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"
    ]
}

response = httpx.post("http://localhost:8000/classify", json=request_data)
result = response.json()

print(f"Classification: {result['results'][0]['matches'][0]['label']}")
print(f"Confidence: {result['results'][0]['matches'][0]['confidence']:.2%}")
```

### Classification with Resource Suggestions

```python
# Combined classification and resource suggestions
request_data = {
    "text": "Interactive group work for mathematics problem solving",
    "mode": "skos",
    "vocabulary_sources": [],  # Use default vocabularies
    "resource_suggestion": True
}

response = httpx.post("http://localhost:8000/classify", json=request_data)
result = response.json()

# Access classification results
for result in result['results']:
    print(f"Property: {result['property']}")
    for match in result['matches']:
        print(f"  - {match['label']} ({match['confidence']:.2%})")

# Access resource suggestions
suggestions = result['resource_suggestion_fields']
print(f"Focus: {suggestions['focus_type']}")
print(f"Search term: {suggestions['search_term']}")
print(f"Keywords: {', '.join(suggestions['keywords'])}")
```

## Architecture

The API is built with:

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **OpenAI**: Language model integration
- **Loguru**: Structured logging
- **Caching**: TTL-based vocabulary caching

## Getting Started

1. [Installation](getting-started/installation.md) - Set up the API
2. [Configuration](getting-started/configuration.md) - Configure your environment
3. [Quick Start](getting-started/quickstart.md) - Make your first request

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Classify text |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

## Support

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/janschachtschabel/classificationapi).
