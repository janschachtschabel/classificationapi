"""Main FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import scoring
from .api.dependencies import get_classification_service
from .api.errors import (
    ClassificationAPIException,
    classification_api_exception_handler,
    general_exception_handler,
    http_exception_handler,
)
from .api.routes import classify, health
from .core.config import settings
from .core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Application lifespan manager.

    Args:
        app: FastAPI application instance

    Yields:
        None: During application lifetime
    """
    # Startup
    setup_logging()

    yield

    # Shutdown
    service = get_classification_service()
    await service.close()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Metadata Classification API",
        description="""
        ## 🎯 Overview

        The **Metadata Classification API** provides intelligent text analysis and classification services for educational content.

        ### 🚀 Key Features

        - **Text Classification**: Classify text using SKOS vocabularies or custom categories
        - **Quality Scoring**: Evaluate text quality with predefined or custom metrics
        - **Metadata Generation**: Generate descriptive fields and resource suggestions
        - **Batch Processing**: Process multiple texts efficiently

        ### 📋 Available Endpoints

        - **`POST /classify`**: Classify text and generate metadata
        - **`POST /scoring/evaluate`**: Evaluate text quality with detailed scoring
        - **`GET /scoring/metrics`**: List available predefined metrics
        - **`GET /health`**: Check API health and dependencies

        ### 🔧 Quick Start

        1. **Classify Text**:
        ```bash
        curl -X POST "http://localhost:8000/classify" \
             -H "Content-Type: application/json" \
             -d '{"text": "Machine learning in education", "mode": "skos"}'
        ```

        2. **Score Text Quality**:
        ```bash
        curl -X POST "http://localhost:8000/scoring/evaluate" \
             -H "Content-Type: application/json" \
             -d '{"text": "Educational content example", "predefined_metrics": ["sachrichtigkeit"]}'
        ```

        ### 📚 Documentation

        - **Interactive Testing**: Use this Swagger UI to test endpoints
        - **Alternative View**: Visit `/redoc` for a clean, readable API reference
        - **Complete Guide**: Full documentation with examples and deployment guides available separately

        ### ⚙️ Configuration

        - **OpenAI API Key**: Required for AI-powered classification and scoring
        - **Environment Variables**: Configure via `.env` file or environment variables
        - **Rate Limits**: Inherits OpenAI's rate limiting policies

        ### 🔒 Authentication

        Currently uses OpenAI API key authentication. Configure your key in environment variables:
        ```bash
        export OPENAI_API_KEY="your-api-key-here"
        ```
        """,
        version=__version__,
        contact={
            "name": "Classification API Team",
            "url": "https://github.com/your-org/classification-api",
            "email": "support@example.com",
        },
        license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    app.add_exception_handler(ClassificationAPIException, classification_api_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, general_exception_handler)

    # Include routers
    app.include_router(health.router)
    app.include_router(classify.router)
    app.include_router(scoring.router)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
