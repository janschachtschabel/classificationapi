# Installation

This guide will help you install and set up the Classification API.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/example/classificationapi.git
cd classificationapi
```

### 2. Install Dependencies with uv

```bash
# Install the package and dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### 3. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4.1-mini

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 4. Verify Installation

Test that everything is working:

```bash
# Run tests
uv run pytest

# Start the API
uv run python -m src.main
```

The API should be available at `http://localhost:8000`.

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the image
docker build -t classificationapi .

# Run the container
docker run -p 8000:8000 --env-file .env classificationapi
```

## Development Setup

For development work, install additional tools:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run code quality checks
uv run ruff check src/
uv run mypy src/
```

## Verification

Once installed, verify the API is working:

1. **Health Check**: Visit `http://localhost:8000/health`
2. **API Documentation**: Visit `http://localhost:8000/docs`
3. **Test Classification**: Make a test request (see [Quick Start](quickstart.md))

## Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
```bash
# Make sure you're running from the project root
cd classificationapi
uv run python -m src.main
```

**OpenAI API Key Error**
```bash
# Verify your API key is set correctly
echo $OPENAI_API_KEY
```

**Port Already in Use**
```bash
# Change the port in .env file
API_PORT=8001
```

### Getting Help

If you encounter issues:

1. Check the [Configuration](configuration.md) guide
2. Review the logs for error messages
3. Open an issue on [GitHub](https://github.com/example/classificationapi/issues)
