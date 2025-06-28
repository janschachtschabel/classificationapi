"""Custom API exceptions and error handlers."""

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger


class ClassificationAPIException(Exception):
    """Base exception for Classification API."""

    def __init__(self, message: str, status_code: int = 500, details: dict[str, Any] | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ClassificationAPIException):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


class VocabularyFetchError(ClassificationAPIException):
    """Raised when vocabulary fetching fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=422, details=details)


class OpenAIError(ClassificationAPIException):
    """Raised when OpenAI API calls fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=502, details=details)


class RateLimitError(ClassificationAPIException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None):
        super().__init__(message, status_code=429, details=details)


class ConfigurationError(ClassificationAPIException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


async def classification_api_exception_handler(request: Request, exc: ClassificationAPIException) -> JSONResponse:
    """Handle custom API exceptions."""
    try:
        logger.error(f"API Exception on {request.method} {request.url.path}: {exc.message} (status: {exc.status_code})")
    except Exception:
        # Fallback if logging fails
        pass

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "details": exc.details,
            }
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    try:
        logger.warning(
            f"HTTP Exception on {request.method} {request.url.path}: {exc.detail} (status: {exc.status_code})"
        )
    except Exception:
        # Fallback if logging fails
        pass

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "HTTPException",
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    try:
        logger.error(f"Unexpected {exc.__class__.__name__} on {request.method} {request.url.path}: {str(exc)}")
    except Exception:
        # Fallback if logging fails - write to stderr directly
        import sys

        print(f"CRITICAL ERROR: {exc.__class__.__name__}: {str(exc)}", file=sys.stderr)

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "InternalServerError",
            }
        },
    )
