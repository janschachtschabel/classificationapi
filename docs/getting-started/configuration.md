# Configuration Guide

Comprehensive configuration options for the Classification API.

## Environment Variables

The API uses environment variables for configuration. Create a `.env` file in the project root:

### Required Settings

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Settings

```env
# OpenAI Model Settings
OPENAI_DEFAULT_MODEL=gpt-4.1-mini          # Default: gpt-4.1-mini
OPENAI_DEFAULT_TEMPERATURE=0.1             # Default: 0.1
OPENAI_DEFAULT_MAX_TOKENS=4000             # Default: 4000
OPENAI_TIMEOUT_SECONDS=60                  # Default: 60

# API Server Settings
HOST=0.0.0.0                               # Default: 0.0.0.0
PORT=8000                                  # Default: 8000
RELOAD=false                               # Default: false

# Logging Settings
LOG_LEVEL=INFO                             # Default: INFO
LOG_FORMAT=json                            # Default: json

# Cache Settings
VOCABULARY_CACHE_TTL=3600                  # Default: 3600 (1 hour)
VOCABULARY_CACHE_SIZE=100                  # Default: 100

# Request Limits
MAX_TEXT_LENGTH=10000                      # Default: 10000
MAX_CATEGORIES=50                          # Default: 50
MAX_VOCABULARIES=10                        # Default: 10
```

## Configuration Classes

### Settings Class

The main configuration is handled by the `Settings` class in `src/core/config.py`:

```python
from src.core.config import get_settings

settings = get_settings()
print(f"Using model: {settings.openai_default_model}")
```

### Available Models

Supported OpenAI models:
- `gpt-4.1-mini` (recommended, default)
- `gpt-4o-mini`
- `gpt-4`
- `gpt-3.5-turbo`

## Logging Configuration

### Log Levels
- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Log Formats
- `json`: Structured JSON logging (default)
- `text`: Human-readable text format

Example log configuration:
```env
LOG_LEVEL=DEBUG
LOG_FORMAT=text
```

## Performance Tuning

### OpenAI Settings
```env
# Faster responses, less creative
OPENAI_DEFAULT_TEMPERATURE=0.1

# Longer responses
OPENAI_DEFAULT_MAX_TOKENS=6000

# Longer timeout for complex requests
OPENAI_TIMEOUT_SECONDS=120
```

### Caching Settings
```env
# Cache vocabularies for 2 hours
VOCABULARY_CACHE_TTL=7200

# Increase cache size for high-traffic scenarios
VOCABULARY_CACHE_SIZE=500
```

### Request Limits
```env
# Allow longer texts
MAX_TEXT_LENGTH=20000

# Allow more categories
MAX_CATEGORIES=100
```

## Production Configuration

### Security
- Never commit `.env` files to version control
- Use environment-specific configuration files
- Rotate API keys regularly
- Use HTTPS in production

### Performance
```env
# Production settings
OPENAI_DEFAULT_MODEL=gpt-4.1-mini
OPENAI_DEFAULT_TEMPERATURE=0.1
OPENAI_DEFAULT_MAX_TOKENS=4000
VOCABULARY_CACHE_TTL=3600
VOCABULARY_CACHE_SIZE=1000
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Monitoring
```env
# Enable detailed logging for monitoring
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Docker Configuration

When using Docker, pass environment variables:

```bash
docker run -e OPENAI_API_KEY=your_key \
           -e OPENAI_DEFAULT_MODEL=gpt-4.1-mini \
           -p 8000:8000 \
           classificationapi
```

## Validation

The API validates configuration on startup. Check logs for any configuration errors:

```bash
uv run python -m src.main
```

Look for configuration validation messages in the startup logs.
