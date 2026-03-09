"""
Simple integration test to verify the SKU router is properly configured.
This test doesn't require a running database - it just verifies the router setup.
"""

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add the enumeration-api directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from main import app
from models.sku import SKU, SearchCriteria, BatchImportRequest, BatchImportResult


def test_router_is_registered():
    """Test that the SKU and MIX routers are registered with the app."""
    # Get all routes from the app
    routes = [route.path for route in app.routes]
    
    # Check that SKU endpoints are registered
    assert "/skus/health" in routes, "Health endpoint not registered"
    assert "/skus/{trade_number}" in routes, "Get SKU endpoint not registered"
    assert "/skus/search" in routes, "Search endpoint not registered"
    assert "/skus/batch" in routes, "Batch import endpoint not registered"
    assert "/skus/export" in routes, "Export endpoint not registered"
    
    # Check that MIX endpoints are registered
    assert "/mixes/health" in routes, "Health endpoint not registered"
    assert "/mixes/{mix_id}" in routes, "Get MIX endpoint not registered"
    assert "/mixes/search" in routes, "Search MIX endpoint not registered"

    print("✓ All SKU and MIX endpoints are registered")


def test_health_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/skus/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    
    print("✓ Health endpoint works correctly")


def test_openapi_schema_generated():
    """Test that OpenAPI schema is generated with SKU and MIX endpoints."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    
    # Check that SKU endpoints are in the schema
    paths = schema.get("paths", {})
    assert "/skus/health" in paths, "Health endpoint not in OpenAPI schema"
    assert "/skus/{trade_number}" in paths, "Get SKU endpoint not in OpenAPI schema"
    assert "/skus/search" in paths, "Search endpoint not in OpenAPI schema"
    assert "/skus/batch" in paths, "Batch import endpoint not in OpenAPI schema"
    assert "/skus/export" in paths, "Export endpoint not in OpenAPI schema"
    
    # Check that MIX endpoints are in the schema
    assert "/mixes/health" in paths, "Health endpoint not in OpenAPI schema"
    assert "/mixes/{mix_id}" in paths, "Get MIX endpoint not in OpenAPI schema"
    assert "/mixes/search" in paths, "Search MIX endpoint not in OpenAPI schema"

    print("✓ OpenAPI schema includes all SKU and MIX endpoints")


if __name__ == "__main__":
    print("Testing SKU Router Integration...")
    print()
    
    try:
        test_router_is_registered()
        test_health_endpoint()
        test_openapi_schema_generated()
        
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
