"""
prediction_service.py
---------------------
Thin service wrapper around predictive_maintenance.predict() and
explainability.explain_sample() so the backend can call them without
importing src/ paths directly.
"""

import sys
from pathlib import Path

# Add src/ to path so we can import project modules
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import predict, load_final_model
from explainability import explain_sample

# Eagerly check the model is loadable at service startup
_model_checked = False


def check_model_available() -> bool:
    """Return True if the final model artefacts exist on disk."""
    try:
        load_final_model()
        return True
    except FileNotFoundError:
        return False


def run_prediction(sensor_data: dict) -> dict:
    """
    Predict normalised RUL and derive tool condition.

    Returns
    -------
    {"predicted_rul": float, "tool_condition": str}
    """
    return predict(sensor_data)


def run_explanation(sensor_data: dict, top_n: int = 5) -> list:
    """
    Return top-N SHAP feature contributions for a single sample.

    Returns
    -------
    list[dict]  — see explainability.explain_sample() for schema
    """
    return explain_sample(sensor_data, top_n=top_n)
