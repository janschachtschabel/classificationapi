# API Overview

The Classification API provides three main endpoints for text processing and analysis.

## Base URL

```
http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Classify text |
| `/scoring/evaluate` | POST | Evaluate text quality |
| `/scoring/metrics` | GET | List available metrics |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

## Authentication

The API uses OpenAI's services internally. Configure your OpenAI API key in the environment variables.

## Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check API status and dependencies
- **Authentication**: None required

### 2. Text Classification
- **Endpoint**: `POST /classify`
- **Purpose**: Classify text, generate metadata, and suggest resources
- **Features**:
  - SKOS vocabulary classification
  - Custom category classification
  - Descriptive fields generation
  - Resource suggestions
  - Combined processing modes

### 3. Text Scoring
- **Endpoint**: `POST /scoring/evaluate`
- **Purpose**: Evaluate text quality using predefined or custom metrics
- **Features**:
  - Predefined metrics (sachrichtigkeit, neutralitaet, aktualitaet)
  - Custom evaluation criteria
  - Detailed scoring with reasoning
  - Improvement suggestions

### 4. Available Metrics
- **Endpoint**: `GET /scoring/metrics`
- **Purpose**: List available predefined scoring metrics
- **Authentication**: None required

## Request/Response Format

All endpoints use JSON format:

```json
{
  "Content-Type": "application/json"
}
```

## Error Handling

The API returns structured error responses:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation error)
- `422`: Unprocessable Entity (schema validation error)
- `500`: Internal Server Error

## Rate Limits

The API inherits OpenAI's rate limits. Monitor your usage through OpenAI's dashboard.

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Processing Modes

### Classification Modes
1. **SKOS Mode**: Uses predefined SKOS vocabularies
2. **Custom Mode**: Uses user-defined categories

### Processing Options
- **Classification Only**: Basic text classification
- **With Descriptive Fields**: Adds metadata generation
- **With Resource Suggestions**: Adds educational resource filtering
- **Combined**: All features together

## Response Structure

All successful responses include:
- `classification_id`: Unique identifier
- `status`: Processing status
- `results`: Classification results (if requested)
- `descriptive_fields`: Generated metadata (if requested)
- `resource_suggestion_fields`: Resource suggestions (if requested)

## Next Steps

- [Health Check API](health.md)
- [Classification API](classification.md)
- [Scoring API](scoring.md)
- [Error Handling](errors.md)
