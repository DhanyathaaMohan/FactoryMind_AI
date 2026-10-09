"""
test_backend.py
---------------
Integration tests for all FastAPI endpoints using starlette TestClient.

Tests
-----
GET  /health
POST /predict
POST /search
POST /ask
"""

import sys
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ #
# Skip if final model artefacts are missing                           #
# ------------------------------------------------------------------ #

MODEL_PATH   = ROOT / "models" / "rul_random_forest_final.pkl"
FEATURE_PATH = ROOT / "models" / "rul_feature_names.json"

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    pytest.skip(
        "Final model artefacts not found — run predictive_maintenance.py first.",
        allow_module_level=True,
    )


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def feature_names():
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def zero_sensor_data(feature_names):
    return {f: 0.0 for f in feature_names}


# ------------------------------------------------------------------ #
# GET /health                                                          #
# ------------------------------------------------------------------ #

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_has_model_loaded_field(self, client):
        r = client.get("/health")
        assert "model_loaded" in r.json()

    def test_health_model_loaded_is_true(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


# ------------------------------------------------------------------ #
# POST /predict                                                         #
# ------------------------------------------------------------------ #

class TestPredict:

    def test_predict_returns_200(self, client, zero_sensor_data):
        r = client.post("/predict", json={"sensor_data": zero_sensor_data})
        assert r.status_code == 200, r.text

    def test_predict_has_predicted_rul(self, client, zero_sensor_data):
        r = client.post("/predict", json={"sensor_data": zero_sensor_data})
        assert "predicted_rul" in r.json()

    def test_predict_rul_in_range(self, client, zero_sensor_data):
        r = client.post("/predict", json={"sensor_data": zero_sensor_data})
        rul = r.json()["predicted_rul"]
        assert 0.0 <= rul <= 1.0, f"RUL out of range: {rul}"

    def test_predict_has_tool_condition(self, client, zero_sensor_data):
        r = client.post("/predict", json={"sensor_data": zero_sensor_data})
        assert "tool_condition" in r.json()

    def test_predict_tool_condition_is_valid(self, client, zero_sensor_data):
        r = client.post("/predict", json={"sensor_data": zero_sensor_data})
        valid = {"Healthy", "Degrading", "Worn", "Critical"}
        assert r.json()["tool_condition"] in valid

    def test_predict_rejects_empty_sensor_data(self, client):
        r = client.post("/predict", json={"sensor_data": {}})
        # Should return 422 (validation/missing features) or 503
        assert r.status_code in (422, 503)

    def test_predict_rejects_missing_body(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422


# ------------------------------------------------------------------ #
# POST /search                                                          #
# ------------------------------------------------------------------ #

SEARCH_QUERIES = [
    "what causes excessive spindle vibration",
    "spindle bearing noise and overheating",
    "what should I inspect when spindle vibration increases",
    "spindle lubrication problem",
    "toolholder causing chatter",
]


class TestSearch:

    @pytest.mark.parametrize("query", SEARCH_QUERIES)
    def test_search_returns_200(self, client, query):
        r = client.post("/search", json={"query": query})
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("query", SEARCH_QUERIES)
    def test_search_returns_results_list(self, client, query):
        r = client.post("/search", json={"query": query})
        assert "results" in r.json()
        assert isinstance(r.json()["results"], list)

    @pytest.mark.parametrize("query", SEARCH_QUERIES)
    def test_search_results_non_empty(self, client, query):
        r = client.post("/search", json={"query": query})
        assert len(r.json()["results"]) >= 1, f"No results for: {query!r}"

    @pytest.mark.parametrize("query", SEARCH_QUERIES)
    def test_search_result_has_source_and_page(self, client, query):
        r = client.post("/search", json={"query": query})
        for result in r.json()["results"]:
            assert "source" in result
            assert "page" in result

    @pytest.mark.parametrize("query", SEARCH_QUERIES)
    def test_search_result_has_hybrid_score(self, client, query):
        r = client.post("/search", json={"query": query})
        for result in r.json()["results"]:
            assert "hybrid_score" in result
            assert result["hybrid_score"] >= 0

    def test_search_rejects_empty_query(self, client):
        r = client.post("/search", json={"query": ""})
        assert r.status_code == 422

    def test_search_respects_top_k(self, client):
        r = client.post("/search", json={"query": "spindle vibration", "top_k": 2})
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 2


# ------------------------------------------------------------------ #
# POST /ask                                                             #
# ------------------------------------------------------------------ #

class TestAsk:

    def test_ask_returns_200(self, client):
        r = client.post("/ask", json={"query": "what causes spindle vibration"})
        assert r.status_code == 200, r.text

    def test_ask_has_maintenance_answer(self, client):
        r = client.post("/ask", json={"query": "spindle bearing noise"})
        assert "maintenance_answer" in r.json()
        assert len(r.json()["maintenance_answer"]) > 0

    def test_ask_has_retrieved_sources(self, client):
        r = client.post("/ask", json={"query": "spindle lubrication problem"})
        assert "retrieved_sources" in r.json()
        assert isinstance(r.json()["retrieved_sources"], list)

    def test_ask_with_rul_returns_tool_condition(self, client):
        r = client.post("/ask", json={
            "query": "spindle vibration",
            "predicted_rul": 0.3,
        })
        assert r.status_code == 200
        assert r.json()["tool_condition"] is not None

    def test_ask_rejects_empty_query(self, client):
        r = client.post("/ask", json={"query": ""})
        assert r.status_code == 422

    def test_ask_rul_out_of_range_rejected(self, client):
        r = client.post("/ask", json={"query": "spindle", "predicted_rul": 1.5})
        assert r.status_code == 422


# --------------------------------------------------
# /samples (new endpoint tests)                       #
# --------------------------------------------------

class TestSamples:

    def test_samples_returns_200(self, client):
        r = client.get("/samples")
        assert r.status_code == 200, r.text

    def test_samples_has_samples_list(self, client):
        r = client.get("/samples")
        assert "samples" in r.json()
        assert isinstance(r.json()["samples"], list)
        assert len(r.json()["samples"]) > 0

    def test_sample_has_required_fields(self, client):
        r = client.get("/samples")
        samples = r.json()["samples"]
        for sample in samples:
            assert "sample_index" in sample
            assert "tool_index" in sample
            assert "cycle" in sample

    def test_sample_index_is_integer(self, client):
        r = client.get("/samples")
        samples = r.json()["samples"]
        for sample in samples:
            assert isinstance(sample["sample_index"], int)

    def test_tool_index_is_integer(self, client):
        r = client.get("/samples")
        samples = r.json()["samples"]
        for sample in samples:
            assert isinstance(sample["tool_index"], int)

    def test_cycle_is_integer(self, client):
        r = client.get("/samples")
        samples = r.json()["samples"]
        for sample in samples:
            assert isinstance(sample["cycle"], int)

    def test_analyze_and_ask_with_sample_index(self, client):
        """Full pipeline with sample_index should work."""
        r = client.post("/analyze-and-ask", json={
            "query": "spindle vibration",
            "sample_index": 0,
        })
        assert r.status_code == 200, r.text
        assert "predicted_rul" in r.json()
        assert "tool_condition" in r.json()
        assert "top_shap_features" in r.json()
        assert "maintenance_answer" in r.json()
        assert len(r.json()["maintenance_answer"]) > 0

    def test_analyze_and_ask_rejects_missing_sample_and_sensor(self, client):
        """Request without sample_index or sensor_data should be rejected."""
        r = client.post("/analyze-and-ask", json={
            "query": "spindle vibration",
        })
        assert r.status_code == 422

    def test_analyze_and_ask_rejects_both_sample_and_sensor(self, client):
        """Request with both sample_index and sensor_data should be rejected."""
        r = client.post("/analyze-and-ask", json={
            "query": "spindle vibration",
            "sample_index": 0,
            "sensor_data": {"Accelerometer - Spindle -X - std": 0.012},
        })
        assert r.status_code == 422

    def test_sample_not_found_returns_404(self, client):
        """Non-existent sample index should return 404."""
        r = client.get("/samples/999999")
        assert r.status_code == 404
