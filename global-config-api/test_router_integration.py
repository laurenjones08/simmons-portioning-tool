"""
Simple integration test to verify the Config router is properly configured.
This test doesn't require a running database - it just verifies the router setup.
"""

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add the global-config-api directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from main import app
from models.config import Config, ConfigUpdate, BatchUpdateRequest, BatchUpdateResult


def test_router_is_registered():
    """Test that the Config endpoints are registered with the app."""
    # Get all routes from the app
    routes = [route.path for route in app.routes]
    
    # Check that Config endpoints are registered
    assert "/config/health" in routes, "Health endpoint not registered"
    assert "/config/{key}" in routes, "Get/Update config endpoint not registered"
    assert "/config" in routes, "Get all configs endpoint not registered"
    assert "/config/batch" in routes, "Batch update endpoint not registered"
    assert "/lines" in routes, "List/create lines endpoint not registered"
    assert "/lines/active" in routes, "Active lines endpoint not registered"
    assert "/lines/{line_id}" in routes, "Line detail endpoint not registered"
    
    print("✓ All Config endpoints are registered")


def test_health_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/config/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    
    print("✓ Health endpoint works correctly")


def test_openapi_schema_generated():
    """Test that OpenAPI schema is generated with Config endpoints."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    
    # Check that Config endpoints are in the schema
    paths = schema.get("paths", {})
    assert "/config/health" in paths, "Health endpoint not in OpenAPI schema"
    assert "/config/{key}" in paths, "Get/Update config endpoint not in OpenAPI schema"
    assert "/config" in paths, "Get all configs endpoint not in OpenAPI schema"
    assert "/config/batch" in paths, "Batch update endpoint not in OpenAPI schema"
    assert "/lines" in paths, "Lines endpoint not in OpenAPI schema"
    assert "/lines/active" in paths, "Active lines endpoint not in OpenAPI schema"
    assert "/lines/{line_id}" in paths, "Line detail endpoint not in OpenAPI schema"
    
    print("✓ OpenAPI schema includes all Config endpoints")


def test_endpoint_methods():
    """Test that endpoints have the correct HTTP methods."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema.get("paths", {})
    
    # Check HTTP methods for each endpoint
    assert "get" in paths["/config/health"], "Health endpoint should support GET"
    assert "get" in paths["/config/{key}"], "Get config endpoint should support GET"
    assert "put" in paths["/config/{key}"], "Update config endpoint should support PUT"
    assert "get" in paths["/config"], "Get all configs endpoint should support GET"
    assert "post" in paths["/config/batch"], "Batch update endpoint should support POST"
    assert "get" in paths["/lines"], "Lines endpoint should support GET"
    assert "post" in paths["/lines"], "Lines endpoint should support POST"
    assert "get" in paths["/lines/active"], "Active lines endpoint should support GET"
    assert "get" in paths["/lines/{line_id}"], "Line detail endpoint should support GET"
    assert "put" in paths["/lines/{line_id}"], "Line detail endpoint should support PUT"
    assert "delete" in paths["/lines/{line_id}"], "Line detail endpoint should support DELETE"
    
    print("✓ All endpoints have correct HTTP methods")


if __name__ == "__main__":
    print("Testing Config Router Integration...")
    print()
    
    try:
        test_router_is_registered()
        test_health_endpoint()
        test_openapi_schema_generated()
        test_endpoint_methods()
        
        print()
        print("=" * 50)
        print("All router integration tests passed! ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"Test failed: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"Unexpected error: {e}")
        print("=" * 50)
        sys.exit(1)
