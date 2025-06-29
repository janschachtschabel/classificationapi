"""Health check endpoint."""

from datetime import UTC, datetime

from fastapi import APIRouter
from loguru import logger

from ... import __version__
from ...schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Check and Status Monitoring",
    description="""
    **Monitor the health and availability of the Metadata Classification API.**

    ## 🟢 What This Endpoint Provides

    - **Service Status**: Overall API health (healthy/unhealthy)
    - **Version Information**: Current API version for compatibility checks
    - **Timestamp**: Current server time in UTC
    - **Uptime Monitoring**: How long the service has been running

    ## 🔍 Use Cases

    ### **Load Balancer Health Checks**
    ```bash
    # Check if service is ready to receive traffic
    curl -f http://localhost:8000/health
    ```

    ### **Monitoring & Alerting**
    ```bash
    # Automated monitoring scripts
    if curl -s http://localhost:8000/health | jq -r '.status' == "healthy"; then
        echo "API is healthy"
    else
        echo "API needs attention"
    fi
    ```

    ### **Deployment Verification**
    ```bash
    # Verify deployment after updates
    curl http://localhost:8000/health | jq '.version'
    ```

    ## 📋 Response Format

    **Healthy Response (200 OK)**:
    ```json
    {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2024-01-15T10:30:00Z",
        "uptime": 3600.5
    }
    ```

    **Unhealthy Response (503 Service Unavailable)**:
    ```json
    {
        "status": "unhealthy",
        "version": "1.0.0",
        "timestamp": "2024-01-15T10:30:00Z",
        "error": "OpenAI API connection failed"
    }
    ```

    ## ⚙️ Integration Notes

    - **No Authentication Required**: Public endpoint for monitoring
    - **Fast Response**: Optimized for frequent health checks
    - **Standard HTTP Codes**: 200 for healthy, 503 for unhealthy
    - **JSON Format**: Machine-readable response for automation

    ## 📊 Monitoring Best Practices

    - **Check Frequency**: Every 30-60 seconds for load balancers
    - **Timeout**: Set 5-10 second timeout for health checks
    - **Alerting**: Alert on consecutive failures (3+ failures)
    - **Logging**: Log health check failures for debugging
    """,
)
async def health_check() -> HealthResponse:
    """
    Perform a health check on the API.

    Returns:
        HealthResponse: Current health status, version, and timestamp
    """
    logger.info("Health check requested")

    return HealthResponse(status="healthy", version=__version__, timestamp=datetime.now(UTC))
