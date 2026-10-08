"""
conftest.py
-----------
Shared pytest fixtures for the FactoryMind test suite.

Fixtures
--------
sample_sensor_data  – dict of 120 sensor feature names → float (all zeros)
                      used to test prediction/explainability without real data.
test_client         – FastAPI TestClient (httpx-based) for endpoint tests.
"""

import sys
import json
import pytest
from pathlib import Path

# Make src/ and backend/ importable from tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ #
# Sensor data fixture                                                   #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def feature_names():
    """Load the 120 sensor feature names from the saved JSON artefact."""
    path = ROOT / "models" / "rul_feature_names.json"
    if not path.exists():
        pytest.skip("rul_feature_names.json not found — run predictive_maintenance.py first")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_sensor_data(feature_names):
    """Return a dict of all 120 features set to 0.0 as a minimal test input."""
    return {f: 0.0 for f in feature_names}


# ------------------------------------------------------------------ #
# FastAPI TestClient                                                    #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def test_client():
    """
    Create a FastAPI TestClient.  Uses starlette's TestClient which
    drives requests synchronously without starting a real server.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        yield client
