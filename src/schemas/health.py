"""Health check schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response model."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    status: str = Field(description="Health status of the API")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Current timestamp")
