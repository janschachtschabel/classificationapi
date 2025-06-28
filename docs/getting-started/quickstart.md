# Quick Start Guide

Get up and running with the Classification API in minutes.

## Prerequisites

- Python 3.13+
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) package manager

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/janschachtschabel/classificationapi.git
cd classificationapi

# Install dependencies
uv sync
```

## 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4.1-mini
OPENAI_DEFAULT_TEMPERATURE=0.1
OPENAI_DEFAULT_MAX_TOKENS=4000
OPENAI_TIMEOUT_SECONDS=60
```

## 3. Start the API

```bash
# Run the development server
uv run python -m src.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 4. First API Call

### Health Check
```bash
curl http://localhost:8000/health
```

### Text Classification
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis is the process by which plants convert sunlight into energy",
    "mode": "skos"
  }'
```

### Text Scoring
```bash
curl -X POST "http://localhost:8000/scoring/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a well-structured educational text about biology.",
    "predefined_metrics": ["sachrichtigkeit"]
  }'
```

### Resource Suggestions
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I need materials for teaching photosynthesis to 7th grade students",
    "resource_suggestion": true
  }'
```

## 5. Interactive Documentation

Visit http://localhost:8000/docs to explore the API interactively with Swagger UI.

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [API Reference](../api/overview.md) - Complete API documentation
- [Examples](../examples/skos.md) - More usage examples
