# Error Handling

Comprehensive guide to API error responses and handling strategies.

## Error Response Format

All API errors follow a consistent JSON structure:

```json
{
  "detail": "Human-readable error description",
  "error_code": "MACHINE_READABLE_ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "request_id": "uuid-string"
}
```

## HTTP Status Codes

### 400 Bad Request
Invalid request parameters or malformed data.

```json
{
  "detail": "Invalid request format",
  "error_code": "BAD_REQUEST",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 422 Unprocessable Entity
Request validation failed (Pydantic validation errors).

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {}
    },
    {
      "type": "string_too_long",
      "loc": ["body", "text"],
      "msg": "String should have at most 10000 characters",
      "input": "very long text..."
    }
  ]
}
```

### 500 Internal Server Error
Server-side errors including external service failures.

```json
{
  "detail": "OpenAI API temporarily unavailable",
  "error_code": "OPENAI_SERVICE_ERROR",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## Error Codes

### Classification API Errors

| Error Code | Description | HTTP Status |
|------------|-------------|-------------|
| `CLASSIFICATION_ERROR` | General classification failure | 500 |
| `VOCABULARY_FETCH_ERROR` | Failed to fetch SKOS vocabularies | 500 |
| `OPENAI_API_ERROR` | OpenAI service error | 500 |
| `OPENAI_RATE_LIMIT` | OpenAI rate limit exceeded | 429 |
| `OPENAI_TIMEOUT` | OpenAI request timeout | 504 |
| `INVALID_VOCABULARY_URL` | Invalid vocabulary URL provided | 400 |
| `TEXT_TOO_LONG` | Text exceeds maximum length | 422 |
| `NO_CATEGORIES_PROVIDED` | No classification categories specified | 422 |

### Scoring API Errors

| Error Code | Description | HTTP Status |
|------------|-------------|-------------|
| `SCORING_ERROR` | General scoring failure | 500 |
| `NO_METRICS_SPECIFIED` | No evaluation metrics provided | 422 |
| `INVALID_METRIC` | Unknown predefined metric | 400 |
| `CUSTOM_METRIC_ERROR` | Invalid custom metric definition | 422 |
| `EVALUATION_TIMEOUT` | Evaluation process timeout | 504 |

### General API Errors

| Error Code | Description | HTTP Status |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `CONFIGURATION_ERROR` | API configuration issue | 500 |
| `DEPENDENCY_ERROR` | External dependency failure | 500 |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded | 429 |

## Common Error Scenarios

### 1. Missing Required Fields

**Request:**
```json
{
  "mode": "skos"
  // Missing "text" field
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {"mode": "skos"}
    }
  ]
}
```

### 2. Text Too Long

**Request:**
```json
{
  "text": "Very long text exceeding 10,000 characters...",
  "mode": "skos"
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "text"],
      "msg": "String should have at most 10000 characters",
      "input": "Very long text..."
    }
  ]
}
```

### 3. Invalid Vocabulary URL

**Request:**
```json
{
  "text": "Sample text",
  "mode": "skos",
  "vocabulary_sources": ["invalid-url"]
}
```

**Response (400):**
```json
{
  "detail": "Invalid vocabulary URL: invalid-url",
  "error_code": "INVALID_VOCABULARY_URL",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 4. OpenAI API Error

**Response (500):**
```json
{
  "detail": "OpenAI API error: Insufficient quota",
  "error_code": "OPENAI_API_ERROR",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 5. Rate Limit Exceeded

**Response (429):**
```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "error_code": "OPENAI_RATE_LIMIT",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "retry_after": 60
}
```

## Error Handling Best Practices

### 1. Python Example

```python
import requests
from typing import Dict, Any

def classify_text(text: str, mode: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            "http://localhost:8000/classify",
            json={"text": text, "mode": mode},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            # Validation error
            errors = response.json()["detail"]
            print(f"Validation errors: {errors}")
            return None
        elif response.status_code == 429:
            # Rate limit
            retry_after = response.headers.get("Retry-After", 60)
            print(f"Rate limited. Retry after {retry_after} seconds")
            return None
        elif response.status_code == 500:
            # Server error
            error_data = response.json()
            print(f"Server error: {error_data['detail']}")
            return None
        else:
            print(f"Unexpected error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timeout")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 2. JavaScript Example

```javascript
async function classifyText(text, mode) {
  try {
    const response = await fetch('http://localhost:8000/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, mode }),
    });

    if (response.ok) {
      return await response.json();
    }

    const errorData = await response.json();
    
    switch (response.status) {
      case 422:
        console.error('Validation errors:', errorData.detail);
        break;
      case 429:
        console.error('Rate limited. Retry after:', response.headers.get('Retry-After'));
        break;
      case 500:
        console.error('Server error:', errorData.detail);
        break;
      default:
        console.error('Unexpected error:', response.status, errorData);
    }
    
    return null;
  } catch (error) {
    console.error('Network error:', error);
    return null;
  }
}
```

### 3. Retry Logic

```python
import time
import random

def classify_with_retry(text: str, mode: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/classify",
                json={"text": text, "mode": mode}
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            else:
                # Don't retry for other errors
                break
                
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    return None
```

## Monitoring and Logging

### Log Error Patterns

Monitor these error patterns in your logs:
- High rate of 422 errors (validation issues)
- Frequent 429 errors (rate limiting)
- 500 errors with OpenAI codes (service issues)
- Timeout errors (performance issues)

### Alerting

Set up alerts for:
- Error rate > 5%
- OpenAI API errors
- Response time > 30 seconds
- Rate limit warnings

## Troubleshooting

### Common Issues

1. **"Field required" errors**: Check request schema
2. **"String too long" errors**: Validate text length before sending
3. **OpenAI API errors**: Check API key and quota
4. **Timeout errors**: Reduce text length or increase timeout
5. **Rate limit errors**: Implement retry logic with backoff

### Debug Mode

Enable debug logging to see detailed error information:

```env
LOG_LEVEL=DEBUG
```
