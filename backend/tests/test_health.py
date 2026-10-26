"""Tests for service health check and core endpoints."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test GET /api/health returns 200 and expected schema keys."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert data["app_name"] == "BookMind API"


def test_root_endpoint():
    """Test GET / returns online status and navigation links."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert "docs" in data
    assert data["health"] == "/api/health"
