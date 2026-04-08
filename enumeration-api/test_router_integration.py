"""
Simple integration test to verify the SKU router is properly configured.
This test doesn't require a running database - it just verifies the router setup.
"""

from fastapi.testclient import TestClient
import sys
import os

# Add the enumeration-api directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from main import app


def test_router_is_registered():
    """Test that all API routers are registered with the app."""
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

    # Check that MixMetric endpoints are registered
    assert "/metrics/health" in routes, "MixMetric health endpoint not registered"
    assert "/metrics/{metric_id}" in routes, "Get MixMetric endpoint not registered"
    assert "/metrics/search" in routes, "Search MixMetric endpoint not registered"

    # Check that Bucket endpoints are registered
    assert "/buckets/health" in routes, "Bucket health endpoint not registered"
    assert "/buckets/{bucket_id}" in routes, "Get Bucket endpoint not registered"
    assert "/buckets/search" in routes, "Search Bucket endpoint not registered"

    # Check that CutStrategy endpoints are registered
    assert "/cut-strategies/health" in routes, "CutStrategy health endpoint not registered"
    assert "/cut-strategies/{strategy_id}" in routes, "Get CutStrategy endpoint not registered"
    assert "/cut-strategies/search" in routes, "Search CutStrategy endpoint not registered"

    print("✓ All SKU and MIX endpoints are registered")


def test_health_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/skus/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    
    print("✓ Health endpoint works correctly")


def test_openapi_schema_generated():
    """Test that OpenAPI schema is generated with all registered endpoints."""
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

    # Check that MixMetric endpoints are in the schema
    assert "/metrics/health" in paths, "MixMetric health endpoint not in OpenAPI schema"
    assert "/metrics/{metric_id}" in paths, "Get MixMetric endpoint not in OpenAPI schema"
    assert "/metrics/search" in paths, "Search MixMetric endpoint not in OpenAPI schema"

    # Check that Bucket endpoints are in the schema
    assert "/buckets/health" in paths, "Bucket health endpoint not in OpenAPI schema"
    assert "/buckets/{bucket_id}" in paths, "Get Bucket endpoint not in OpenAPI schema"
    assert "/buckets/search" in paths, "Search Bucket endpoint not in OpenAPI schema"

    # Check that CutStrategy endpoints are in the schema
    assert "/cut-strategies/health" in paths, "CutStrategy health endpoint not in OpenAPI schema"
    assert "/cut-strategies/{strategy_id}" in paths, "Get CutStrategy endpoint not in OpenAPI schema"
    assert "/cut-strategies/search" in paths, "Search CutStrategy endpoint not in OpenAPI schema"

    # Verify endpoint-level response examples for cascade delete operations.
    bucket_delete_example = (
        paths["/buckets/{bucket_id}"]["delete"]["responses"]["200"]["content"]["application/json"]["example"]
    )
    assert "warning" in bucket_delete_example
    assert "metricsDeleted" in bucket_delete_example

    cut_strategy_delete_example = (
        paths["/cut-strategies/{strategy_id}"]["delete"]["responses"]["200"]["content"]["application/json"]["example"]
    )
    assert "mixesDeleted" in cut_strategy_delete_example
    assert "metricsDeleted" in cut_strategy_delete_example

    # Verify request model examples exist for create/search payloads.
    schemas = schema.get("components", {}).get("schemas", {})
    assert "example" in schemas["BucketCreate"]
    assert "example" in schemas["BucketSearchCriteria"]
    assert "example" in schemas["CutStrategyCreate"]
    assert "example" in schemas["CutStrategySearchCriteria"]

    print("✓ OpenAPI schema includes all endpoints and expected examples")


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
