"""Pytest configuration for enumeration-worker-api.

Adding this file at the service root ensures pytest and the IDE
treat enumeration-worker-api/ as the Python source root, making
all absolute imports (database, job_service, models.*, etc.) resolve correctly.
"""
import sys
from pathlib import Path

# Ensure the service root is on sys.path for all tests
_service_root = str(Path(__file__).parent)
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)

