# Health Check API

Monitor the API status and dependencies.

## Endpoint

```
GET /health
```

## Description

The health check endpoint provides information about the API status and its dependencies.

## Request

No parameters required.

```bash
curl -X GET "http://localhost:8000/health"
```

## Response

### Success Response (200 OK)

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "version": "1.0.0",
  "dependencies": {
    "openai": "available",
    "database": "not_applicable"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall health status: "healthy" or "unhealthy" |
| `timestamp` | string | ISO 8601 timestamp of the health check |
| `version` | string | API version |
| `dependencies` | object | Status of external dependencies |
| `dependencies.openai` | string | OpenAI API availability |
| `dependencies.database` | string | Database status (not applicable for this API) |

### Status Values

- **healthy**: All systems operational
- **unhealthy**: One or more critical systems unavailable

### Dependency Status Values

- **available**: Service is accessible and responding
- **unavailable**: Service is not accessible
- **not_applicable**: Service is not used by this API

## Error Response

If the API is experiencing issues:

```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "version": "1.0.0",
  "dependencies": {
    "openai": "unavailable",
    "database": "not_applicable"
  },
  "errors": [
    "OpenAI API key not configured",
    "OpenAI API not responding"
  ]
}
```

## Usage Examples

### Basic Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### With Python requests

```python
import requests

response = requests.get("http://localhost:8000/health")
health_data = response.json()

if health_data["status"] == "healthy":
    print("API is healthy")
else:
    print(f"API issues: {health_data.get('errors', [])}")
```

### With JavaScript fetch

```javascript
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => {
    if (data.status === 'healthy') {
      console.log('API is healthy');
    } else {
      console.log('API issues:', data.errors);
    }
  });
```

## Monitoring

Use this endpoint for:
- **Load balancer health checks**
- **Monitoring system integration**
- **Automated deployment validation**
- **Service discovery health verification**

## Response Time

The health check is designed to respond quickly (< 100ms) and doesn't perform expensive operations.

## Security

The health endpoint doesn't expose sensitive information and can be safely used by monitoring systems.
