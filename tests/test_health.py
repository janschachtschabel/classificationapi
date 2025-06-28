"""Tests for health check endpoint."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint returns correct response."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_health_check_response_structure(client: TestClient):
    """Test health check response has correct structure."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    # Check required fields
    required_fields = ["status", "version", "timestamp"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Check field types
    assert isinstance(data["status"], str)
    assert isinstance(data["version"], str)
    assert isinstance(data["timestamp"], str)

    # Check timestamp format (ISO 8601)
    try:
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO 8601 format")
