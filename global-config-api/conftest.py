"""
Pytest configuration and fixtures for the Global Config API tests.

This module provides shared fixtures for all tests, including database connections
and test data generators.
"""

import pytest
import mongomock
import sys
from pathlib import Path

# Add the app directory to Python path for imports
# This allows tests to import modules using absolute imports like "from models.config import Config"
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


@pytest.fixture
def mongodb_database():
    """
    Provide a clean MongoDB database for each test using mongomock.
    
    This fixture creates an in-memory mock MongoDB database for testing.
    Each test gets a fresh database instance, ensuring test isolation.
    
    Yields:
        Database: Mock MongoDB database instance for testing
    """
    # Use mongomock for in-memory MongoDB testing
    client = mongomock.MongoClient()
    
    # Use a dedicated test database
    db = client["test_config_db"]
    
    try:
        # Yield the database to the test
        yield db
    finally:
        # Clean up after the test
        client.close()
