# Classification Endpoint

The `/classify` endpoint is the core functionality of the Classification API, providing zero-shot text classification using OpenAI's language models.

## Endpoint

```
POST /classify
```

## Request Format

### Request Body

```json
{
  "text": "Text to classify",
  "mode": "skos" | "custom",
  "vocabulary_sources": ["https://example.com/vocab1.json"],
  "custom_categories": {
    "property1": ["value1", "value2"],
    "property2": ["valueA", "valueB"]
  },
  "model": "gpt-4.1-mini",
  "temperature": 0.2,
  "max_tokens": 15000
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | ✅ | Text to classify (1-50,000 characters) |
| `mode` | string | ✅ | Classification mode: `"skos"` or `"custom"` |
| `vocabulary_sources` | array | ⚠️ | SKOS vocabulary URLs (required for `skos` mode) |
| `custom_categories` | object | ⚠️ | Custom categories (required for `custom` mode) |
| `model` | string | ❌ | OpenAI model (defaults to configured default) |
| `temperature` | number | ❌ | Temperature 0.0-2.0 (default: 0.2) |
| `max_tokens` | integer | ❌ | Max tokens 1-50,000 (default: 15,000) |

## Response Format

### Success Response (200)

```json
{
  "classification_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": [
    {
      "property": "Subject",
      "matches": [
        {
          "id": "http://example.com/math",
          "label": "Mathematics",
          "confidence": 0.95,
          "explanation": "The text discusses mathematical concepts and formulas"
        }
      ]
    }
  ],
  "metadata": {
    "model": "gpt-4.1-mini",
    "model_settings": {
      "temperature": 0.2,
      "max_tokens": 15000
    },
    "timestamp": "2025-06-24T11:05:00Z",
    "processing_time_ms": 1234
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `classification_id` | UUID | Unique identifier for this classification |
| `status` | string | Classification status (`"completed"`) |
| `results` | array | Classification results for each property |
| `results[].property` | string | Name of the classified property |
| `results[].matches` | array | Matching values for this property |
| `results[].matches[].id` | string | Unique identifier for the match |
| `results[].matches[].label` | string | Human-readable label |
| `results[].matches[].confidence` | number | Confidence score (0.0-1.0) |
| `results[].matches[].explanation` | string | Explanation for the match |
| `metadata` | object | Request metadata |
| `metadata.model` | string | OpenAI model used |
| `metadata.model_settings` | object | Model configuration used |
| `metadata.timestamp` | string | Completion timestamp (ISO 8601) |
| `metadata.processing_time_ms` | integer | Processing time in milliseconds |

## Classification Modes

### SKOS Mode

Uses SKOS vocabularies from external URLs (e.g., SKOHub):

```json
{
  "text": "This course covers linear algebra and calculus.",
  "mode": "skos",
  "vocabulary_sources": [
    "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"
  ]
}
```

### Custom Mode

Uses your own predefined categories:

```json
{
  "text": "This is an advanced programming course.",
  "mode": "custom",
  "custom_categories": {
    "Subject": ["Programming", "Mathematics", "Science"],
    "Level": ["Beginner", "Intermediate", "Advanced"]
  }
}
```

## Examples

### Basic SKOS Classification

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Introduction to machine learning algorithms and neural networks",
    "mode": "skos",
    "vocabulary_sources": [
      "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json"
    ]
  }'
```

### Custom Categories with Model Settings

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Advanced calculus for engineering students",
    "mode": "custom",
    "custom_categories": {
      "Subject": ["Mathematics", "Engineering", "Physics"],
      "Level": ["Beginner", "Intermediate", "Advanced"]
    },
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 2000
  }'
```

### Python Example

```python
import httpx
import json

# Prepare request
request_data = {
    "text": "This course introduces students to organic chemistry principles",
    "mode": "custom",
    "custom_categories": {
        "Subject": ["Chemistry", "Biology", "Physics", "Mathematics"],
        "Level": ["Introductory", "Intermediate", "Advanced"]
    }
}

# Make request
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/classify",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        for classification in result["results"]:
            print(f"Property: {classification['property']}")
            for match in classification["matches"]:
                print(f"  - {match['label']}: {match['confidence']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
```

## Error Responses

### Validation Error (422)

```json
{
  "error": {
    "message": "vocabulary_sources is required for 'skos' mode",
    "type": "ValidationError",
    "details": {}
  }
}
```

### Vocabulary Fetch Error (422)

```json
{
  "error": {
    "message": "Failed to fetch vocabulary from https://example.com/vocab.json",
    "type": "VocabularyFetchError",
    "details": {
      "url": "https://example.com/vocab.json"
    }
  }
}
```

### OpenAI API Error (502)

```json
{
  "error": {
    "message": "OpenAI API call failed: Rate limit exceeded",
    "type": "OpenAIError",
    "details": {
      "model": "gpt-4.1-mini"
    }
  }
}
```

## Best Practices

1. **Text Length**: Keep text under 10,000 characters for optimal performance
2. **Vocabulary Caching**: Reuse the same vocabulary URLs to benefit from caching
3. **Temperature**: Use lower temperatures (0.1-0.3) for more consistent results
4. **Error Handling**: Always check response status and handle errors gracefully
5. **Rate Limiting**: Be mindful of OpenAI API rate limits

## Supported Models

- `gpt-4o-mini` (fastest, most cost-effective)
- `gpt-4o` (balanced performance)
- `gpt-4.1` (highest quality)
- `gpt-4.1-mini` (recommended default)
- `gpt-4.1-nano` (ultra-fast)
