# Text Scoring API

Evaluate text quality using predefined or custom metrics with detailed scoring and improvement suggestions.

## Endpoints

### 1. Evaluate Text
```
POST /scoring/evaluate
```

### 2. Available Metrics
```
GET /scoring/metrics
```

## Text Evaluation

### Request

```json
{
  "text": "Your text to evaluate",
  "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],
  "custom_metrics": [
    {
      "name": "clarity",
      "description": "Evaluate text clarity and readability",
      "criteria": [
        {
          "name": "sentence_structure",
          "description": "Clear and well-structured sentences",
          "weight": 2.0
        },
        {
          "name": "vocabulary",
          "description": "Appropriate vocabulary for target audience",
          "weight": 1.5
        }
      ]
    }
  ],
  "include_improvements": true
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to evaluate (max 10,000 characters) |
| `predefined_metrics` | array | No | List of predefined metrics to use |
| `custom_metrics` | array | No | Custom evaluation criteria |
| `include_improvements` | boolean | No | Include improvement suggestions (default: false) |

### Predefined Metrics

#### sachrichtigkeit (Factual Accuracy)
Evaluates factual correctness and accuracy:
- Sachliche Richtigkeit (Factual Correctness)
- Klarheit (Clarity)
- Objektivität (Objectivity)
- Relevanz (Relevance)
- Struktur (Structure)
- Sprache (Language)

#### neutralitaet (Neutrality)
Comprehensive neutrality evaluation with 24 criteria:
- Perspective diversity
- Neutral description
- Opinion formation support
- Linguistic neutrality
- Source credibility
- Scientific fairness
- Political balance
- And 17 more detailed criteria

#### aktualitaet (Timeliness)
Evaluates content currency and timeliness:
- Temporal references
- Current research integration
- Legal currency
- Contemporary examples
- Modern terminology

### Response

```json
{
  "evaluation_id": "uuid-string",
  "text_preview": "First 100 characters of evaluated text...",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "results": [
    {
      "metric_name": "sachrichtigkeit",
      "overall_score": 5.5,
      "max_possible_score": 6.0,
      "normalized_score": 0.92,
      "confidence": 0.95,
      "scale_type": "likert_3",
      "criteria_scores": [
        {
          "criterion_name": "sachliche_richtigkeit",
          "score": 3,
          "max_score": 3,
          "weight": 2.0,
          "reasoning": "The content is factually accurate..."
        }
      ],
      "overall_reasoning": "The text demonstrates high factual accuracy...",
      "suggested_improvements": [
        "Consider adding more specific examples",
        "Clarify technical terminology"
      ]
    }
  ],
  "metadata": {
    "total_metrics": 1,
    "processing_time_seconds": 8.5,
    "model_used": "gpt-4.1-mini"
  }
}
```

## Available Metrics Endpoint

### Request

```bash
curl -X GET "http://localhost:8000/scoring/metrics"
```

### Response

```json
{
  "predefined_metrics": [
    {
      "name": "sachrichtigkeit",
      "display_name": "Sachrichtigkeit",
      "description": "Evaluates factual accuracy and correctness",
      "scale_type": "likert_3",
      "criteria_count": 6
    },
    {
      "name": "neutralitaet",
      "display_name": "Neutralität",
      "description": "Comprehensive neutrality evaluation",
      "scale_type": "likert_4",
      "criteria_count": 24
    },
    {
      "name": "aktualitaet",
      "display_name": "Aktualität",
      "description": "Evaluates content timeliness and currency",
      "scale_type": "likert_5",
      "criteria_count": 20
    }
  ]
}
```

## Usage Examples

### Basic Evaluation

```bash
curl -X POST "http://localhost:8000/scoring/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
    "predefined_metrics": ["sachrichtigkeit"]
  }'
```

### Custom Metrics

```bash
curl -X POST "http://localhost:8000/scoring/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Educational content about biology for students.",
    "custom_metrics": [
      {
        "name": "educational_value",
        "description": "Assess educational effectiveness",
        "criteria": [
          {
            "name": "learning_objectives",
            "description": "Clear learning objectives",
            "weight": 2.0
          },
          {
            "name": "engagement",
            "description": "Student engagement potential",
            "weight": 1.5
          }
        ]
      }
    ],
    "include_improvements": true
  }'
```

### Combined Evaluation

```python
import requests

data = {
    "text": "Your educational text here...",
    "predefined_metrics": ["sachrichtigkeit", "neutralitaet"],
    "custom_metrics": [
        {
            "name": "readability",
            "description": "Text readability assessment",
            "criteria": [
                {
                    "name": "sentence_length",
                    "description": "Appropriate sentence length",
                    "weight": 1.0
                }
            ]
        }
    ],
    "include_improvements": True
}

response = requests.post(
    "http://localhost:8000/scoring/evaluate",
    json=data
)

results = response.json()
for result in results["results"]:
    print(f"{result['metric_name']}: {result['normalized_score']:.2%}")
```

## Error Handling

### Validation Errors (422)

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Text length exceeds maximum of 10000 characters",
      "input": "very long text..."
    }
  ]
}
```

### Service Errors (500)

```json
{
  "detail": "OpenAI API error: Rate limit exceeded",
  "error_code": "OPENAI_RATE_LIMIT",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Scale Types

- **binary**: 0 or 1 (not met / met)
- **likert_3**: 1-3 scale (poor / fair / good)
- **likert_4**: 1-4 scale (poor / fair / good / excellent)
- **likert_5**: 1-5 scale (very poor / poor / fair / good / excellent)

## Best Practices

1. **Text Length**: Keep texts under 10,000 characters for optimal performance
2. **Metric Selection**: Choose relevant metrics for your use case
3. **Custom Criteria**: Define clear, specific criteria for custom metrics
4. **Batch Processing**: For multiple texts, make separate requests
5. **Error Handling**: Always handle potential API errors gracefully
