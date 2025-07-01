# Classification API

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modern, production-ready API for zero-shot classification of educational metadata, text quality scoring, descriptive field generation, and intelligent resource suggestions using OpenAI's language models.

## 🚀 Features

### Core Classification
- **Zero-shot Classification**: No training required, works with any text
- **SKOS Support**: Direct integration with SKOHub vocabularies  
- **Custom Categories**: Use your own classification schemes
- **Semantic Filtering**: Cross-encoder model for vocabulary optimization
- **Vocabulary Caching**: High-performance TTL caching system

### Text Quality Scoring
- **Predefined Metrics**: Sachrichtigkeit, Neutralität, Aktualität evaluation
- **Custom Metrics**: Define your own evaluation criteria
- **Detailed Scoring**: Criterion-level scores with reasoning
- **Improvement Suggestions**: Actionable feedback for content enhancement
- **Multi-scale Support**: Binary, Likert 3/4/5 point scales

### Metadata Generation
- **Descriptive Fields**: Auto-generate title, short_title, keywords, description
- **Educational Resource Suggestions**: Intelligent filter suggestions for content databases
- **Learning Phase Detection**: Introduction, deepening, practice, assessment
- **Focus Type Analysis**: Content-based, methodical, target-group analysis

### Technical Excellence
- **Unified API**: Combine all features in single requests or use standalone
- **Type Safety**: Full MyPy compliance with comprehensive type hints
- **Production Ready**: 76% test coverage, comprehensive error handling
- **High Performance**: Async processing with intelligent caching
- **Comprehensive Logging**: Structured logging with request tracing
- **API Documentation**: Auto-generated OpenAPI docs with examples

## 📊 Predefined Evaluation Criteria

The API includes 5 comprehensive, research-based evaluation metrics for educational content quality assessment:

### 🎯 Sachrichtigkeit (Factual Accuracy)
- **Scale**: 5-point Likert scale (1-5)
- **Focus**: Factual correctness, source quality, scientific accuracy
- **Criteria**: 8 weighted criteria including factual correctness (3.0), source quality (2.5), scientific accuracy (2.5), completeness (2.0), objectivity (2.0), ethical correctness (1.5), consistency (1.5), and transparency (1.0)
- **Use Case**: Verify accuracy and reliability of educational content

### ⚖️ Neutralität (Neutrality)
- **Scale**: 4-point Likert scale (1-4)
- **Focus**: Balanced perspective, objectivity, critical thinking promotion
- **Criteria**: 15 weighted criteria covering perspective diversity (3.0), neutral description (2.5), opinion formation support (2.5), political neutrality (2.0), ideological neutrality (2.0), and media literacy promotion (1.5)
- **Use Case**: Ensure balanced, unbiased educational content

### 🕒 Aktualität (Timeliness/Currency)
- **Scale**: 5-point Likert scale (1-5)
- **Focus**: Current information, recent research, up-to-date examples
- **Criteria**: 20 detailed criteria including temporal references (2.0), current research integration (2.5), legal currency (2.0), societal developments (2.0), recent sources (2.5), contemporary examples (2.0), and current terminology (2.0)
- **Use Case**: Assess content freshness and relevance

### 🎓 Didaktik & Methodik (Didactics & Methodology)
- **Scale**: 5-point Likert scale (1-5)
- **Focus**: Educational design, learning objectives, methodological quality
- **Criteria**: Comprehensive pedagogical evaluation including learning objective clarity, methodological appropriateness, differentiation, and assessment alignment
- **Use Case**: Evaluate educational effectiveness and pedagogical quality

### 📝 Sprachliche Verständlichkeit (Linguistic Comprehensibility)
- **Scale**: 5-point Likert scale (1-5)
- **Focus**: Language clarity, readability, target group appropriateness
- **Criteria**: 8 weighted criteria covering clarity (3.0), sentence structure (2.5), terminology explanation (2.0), target group appropriateness (2.5), inclusivity (2.0), readability (2.0), motivation (1.5), and grammar (1.5)
- **Use Case**: Ensure content accessibility and comprehensibility

### 🔧 Using Predefined Metrics

```python
import httpx

# Evaluate text with multiple predefined metrics
response = httpx.post("http://localhost:8000/scoring/evaluate", json={
    "text": "Photosynthese ist der Prozess, bei dem Pflanzen Lichtenergie in chemische Energie umwandeln...",
    "predefined_metrics": ["sachrichtigkeit", "neutralitaet", "sprachliche_verstaendlichkeit"],
    "include_improvements": true
})

result = response.json()
for evaluation in result["evaluations"]:
    print(f"{evaluation['metric_name']}: {evaluation['normalized_score']:.2f}")
    print(f"Reasoning: {evaluation['overall_reasoning']}")
```

### 📋 Available Metrics Endpoint

```python
# Get list of all available predefined metrics
response = httpx.get("http://localhost:8000/scoring/metrics")
metrics = response.json()
print("Available metrics:", [m["name"] for m in metrics["predefined_metrics"]])
```

## 📋 Requirements

- Python 3.13+
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) package manager

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/janschachtschabel/classificationapi.git
cd classificationapi

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the API
uv run python -m src.main
```

The API will be available at `http://localhost:8000`.

### Docker

#### Quick Start with Docker Compose

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Manual Docker Build

```bash
# Build the image
docker build -t classificationapi .

# Run with environment file
docker run -p 8000:8000 --env-file .env classificationapi

# Or run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your_key_here" \
  -e OPENAI_DEFAULT_MODEL="gpt-4o-mini" \
  classificationapi
```

#### Development with Docker

```bash
# Start in development mode (with auto-reload)
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Or use the shorthand (override is loaded automatically)
docker-compose up
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4o-mini
API_PORT=8000
LOG_LEVEL=INFO
```

See [Configuration Guide](docs/getting-started/configuration.md) for all options.

## 🧪 Live Demo

**🚀 Try it now in Google Colab** (no installation required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YSj4ZQhLQWUuOL48EvZjiQcZmwGpkiwR)

Das Notebook enthält:
- Vollständige API-Installation in Google Colab
- Cloudflare Tunnel für öffentlichen Zugriff

## 📖 Usage

## 🚀 API-Endpoints

```
# Klassifikation & Metadaten
POST /classify             # Klassifikation, Metadaten-Generierung, Ressourcen-Vorschläge

# Textqualität & Scoring
POST /scoring/evaluate     # Textqualität bewerten (Sachrichtigkeit, Neutralität, u.a.)
GET  /scoring/metrics      # Verfügbare Bewertungsmetriken abrufen

# System
GET  /health               # Service-Status
GET  /docs                 # Interaktive API-Dokumentation (Swagger UI)
GET  /redoc                # Alternative API-Dokumentation (ReDoc)
```


### 🔗 API Endpoints

The API provides several endpoints for different types of content analysis:

### `/classify` - Unified Classification Endpoint

The main endpoint supporting multiple processing modes that can be combined:

#### **Classification Modes:**
- **`"skos"`** - Classify using SKOS vocabularies (OpenEduHub, custom vocabularies)
- **`"custom"`** - Classify using custom categories defined in the request

#### **Additional Processing Options:**
- **`generate_descriptive_fields: true`** - Generate title, description, keywords
- **`resource_suggestion: true`** - Generate educational resource suggestions
- **`vocabulary_sources: []`** - Specify custom SKOS vocabularies or use defaults

#### **Request Parameters:**
```json
{
  "text": "Your content to analyze",
  "mode": "skos|custom",
  "vocabulary_sources": ["url1", "url2"],  // Optional, defaults to OpenEduHub
  "custom_categories": {                     // Required for custom mode
    "Subject": ["Math", "Science"],
    "Level": ["Beginner", "Advanced"]
  },
  "generate_descriptive_fields": true,       // Optional, default: false
  "resource_suggestion": true                // Optional, default: false
}
```

### `/scoring/evaluate` - Text Quality Evaluation

Evaluate text quality using predefined or custom metrics:

#### **Available Predefined Metrics:**
- `sachrichtigkeit` - Factual accuracy (5-point scale)
- `neutralitaet` - Neutrality and objectivity (4-point scale)
- `aktualitaet` - Timeliness and currency (5-point scale)
- `didaktik_methodik` - Pedagogical quality (5-point scale)
- `sprachliche_verstaendlichkeit` - Linguistic comprehensibility (5-point scale)

#### **Request Parameters:**
```json
{
  "text": "Content to evaluate",
  "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],  // Optional
  "custom_metrics": [                                          // Optional
    {
      "name": "Custom Metric",
      "description": "Evaluate specific aspect",
      "scale": {
        "type": "likert_5",
        "min": 1,
        "max": 5
      },
      "criteria": [
        {
          "name": "Criterion 1",
          "description": "Detailed description",
          "weight": 2.0
        }
      ]
    }
  ],
  "include_improvements": false                               // Optional, default: false
}
```

### `/scoring/metrics` - Available Metrics

Retrieve list of all available predefined evaluation metrics:

```bash
GET /scoring/metrics
```

### `/health` - Health Check

Simple health check endpoint:

```bash
GET /health
```

### **Interactive Documentation**

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing
- **ReDoc**: `http://localhost:8000/redoc` - Clean API documentation

## 🚀 Usage Examples

### Basic Classification

Classify text using SKOS vocabularies:

```python
import requests

response = requests.post("http://localhost:8000/classify", json={
    "text": "This course covers linear algebra, calculus, and statistics.",
    "mode": "skos"
})

result = response.json()
for vocab_result in result['results']:
    for classification in vocab_result['classifications']:
        print(f"Subject: {classification['preferred_label']}")
        print(f"Confidence: {classification['confidence']:.2%}")
```

#### Text Quality Scoring

Evaluate text quality with predefined metrics:

```python
response = requests.post("http://localhost:8000/scoring/evaluate", json={
    "text": "Photosynthesis is the process by which plants convert sunlight into energy.",
    "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],
    "include_improvements": True
})

result = response.json()
for evaluation in result['results']:
    print(f"Metric: {evaluation['metric_name']}")
    print(f"Score: {evaluation['normalized_score']:.2%}")
    print(f"Suggestions: {evaluation['suggested_improvements']}")
```

#### Combined Processing

Get classification, metadata, and resource suggestions in one request:

```python
response = requests.post("http://localhost:8000/classify", json={
    "text": "Advanced Python programming for data science",
    "mode": "custom", 
    "custom_categories": {
        "Subject": ["Programming", "Mathematics", "Science"],
        "Level": ["Beginner", "Intermediate", "Advanced"]
    }
})
```

### Descriptive Metadata Generation

Generate structured metadata fields for content:

```python
response = httpx.post("http://localhost:8000/classify", json={
    "text": "This comprehensive guide explores machine learning algorithms and their applications in modern data science.",
    "generate_descriptive_fields": true
})

result = response.json()
print(f"Title: {result['descriptive_fields']['title']}")
print(f"Keywords: {', '.join(result['descriptive_fields']['keywords'])}")
print(f"Description: {result['descriptive_fields']['description']}")
```

### Combined Classification and Metadata

Combine classification with descriptive metadata generation:

```python
response = httpx.post("http://localhost:8000/classify", json={
    "text": "Introduction to quantum computing and its potential applications",
    "mode": "skos",
    "vocabulary_sources": [
        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"
    ],
    "generate_descriptive_fields": true
})

result = response.json()
# Access both classification results and descriptive fields
classification = result['results'][0]['matches'][0]['label']
title = result['descriptive_fields']['title']
```

### Flexible SKOS Mode

Use SKOS mode with empty vocabularies for descriptive fields only:

```python
response = httpx.post("http://localhost:8000/classify", json={
    "text": "Exploring renewable energy technologies and sustainability",
    "mode": "skos",
    "vocabulary_sources": [],
    "generate_descriptive_fields": true
})
```

### Educational Resource Suggestions

Generate intelligent suggestions for educational content databases:

```python
response = httpx.post("http://localhost:8000/classify", json={
    "text": "Ich suche Materialien für den Biologieunterricht zum Thema Photosynthese für Klasse 7.",
    "resource_suggestion": true
})

result = response.json()
print(f"Focus Type: {result['resource_suggestion_fields']['focus_type']}")
print(f"Learning Phase: {result['resource_suggestion_fields']['learning_phase']}")
print(f"Search Term: {result['resource_suggestion_fields']['search_term']}")
for suggestion in result['resource_suggestion_fields']['filter_suggestions']:
    print(f"Filter: {suggestion['labels']} (Confidence: {suggestion['confidence']:.2f})")
```

### Complete Unified Workflow

Combine all processing modes in a single request:

```python
response = httpx.post("http://localhost:8000/classify", json={
    "text": "Klimawandel verstehen und Lösungen entwickeln für die Oberstufe.",
    "mode": "skos",
    "vocabulary_sources": [
        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"
    ],
    "generate_descriptive_fields": true,
    "resource_suggestion": true
})

result = response.json()
# Access classification results
classification = result['results'][0]['matches'][0]['label']
# Access descriptive metadata
title = result['descriptive_fields']['title']
# Access resource suggestions
focus_type = result['resource_suggestion_fields']['focus_type']
```

## 📚 API Documentation

- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc` 
- **Full Documentation**: [docs/](docs/)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Unified endpoint for classification, metadata generation, and resource suggestions |
| `/scoring/evaluate` | POST | Text quality evaluation with predefined or custom metrics |
| `/scoring/metrics` | GET | List available predefined evaluation metrics |

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_classification.py
```

## 🔍 Development

### Code Quality

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run linting
uv run ruff check src/

# Run type checking  
uv run mypy src/

# Format code
uv run ruff format src/
```

### Project Structure

```
classificationapi/
├── src/                    # Source code
│   ├── api/               # API routes and dependencies
│   ├── core/              # Configuration and logging
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   └── main.py           # FastAPI application
├── tests/                 # Test suite
├── docs/                  # Documentation
└── pyproject.toml        # Project configuration
```

## 🚀 Deployment

### Docker

```dockerfile
# Build production image
docker build -t classificationapi:latest .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  classificationapi:latest
```

### Environment Variables

Production deployment requires:

- `OPENAI_API_KEY`: Your OpenAI API key
- `API_HOST`: Host to bind (default: 0.0.0.0)
- `API_PORT`: Port to bind (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

## 📊 Performance

- **Vocabulary Caching**: TTL-based caching reduces API calls
- **Async Processing**: Non-blocking I/O for better throughput
- **Request Validation**: Fast Pydantic validation
- **Structured Logging**: Efficient JSON logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run full test suite
uv run pytest --cov=src
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/janschachtschabel/classificationapi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/janschachtschabel/classificationapi/discussions)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [OpenAI](https://openai.com/) for the language models
- [Pydantic](https://pydantic.dev/) for data validation
- [uv](https://github.com/astral-sh/uv) for fast Python package management
