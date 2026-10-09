"""
prediction_service.py
---------------------
Thin service wrapper around predictive_maintenance.predict() and
explainability.explain_sample() so the backend can call them without
importing src/ paths directly.
"""

import json
import sys
from pathlib import Path

# Add src/ to path so we can import project modules
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

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


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

def load_dataset():
    """Load the feature and metadata CSV."""
    DATA_PATH = SRC_DIR.parent / "data" / "FeatureAndMetadata_Milling.csv"
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";", header=1)
    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    return df


def get_feature_names():
    """Load the 120 sensor feature names."""
    # Try multiple possible paths for the feature file
    possible_paths = [
        SRC_DIR.parent / "models" / "rul_feature_names.json",
        Path.cwd() / "models" / "rul_feature_names.json",
        Path(__file__).resolve().parent.parent / "models" / "rul_feature_names.json",
    ]
    
    feature_path = None
    for p in possible_paths:
        if p.exists():
            feature_path = p
            break
    
    if not feature_path or not feature_path.exists():
        raise FileNotFoundError(f"Feature names not found. Tried: {[str(p) for p in possible_paths]}")
    
    with open(feature_path, "r", encoding="utf-8") as f:
        result = json.load(f)
        if result is None:
            raise ValueError(f"Feature names JSON is empty: {feature_path}")
        return result


def get_samples_metadata():
    """
    Return lightweight metadata for all dataset samples.

    Returns
    -------
    {"samples": [{"sample_index": int, "tool_index": int, "cycle": int}, ...]}
    """
    df = load_dataset()
    samples = []
    for idx, row in df.iterrows():
        samples.append({
            "sample_index": int(idx),
            "tool_index": int(row["TollIndex"]),
            "cycle": int(row["NumberOfCycle"]),
        })
    return {"samples": samples}


def get_sample_metadata(sample_index: int):
    """
    Return metadata for a specific sample.

    Returns
    -------
    {"sample_index": int, "tool_index": int, "cycle": int}
    """
    df = load_dataset()
    if sample_index < 0 or sample_index >= len(df):
        raise IndexError(f"Sample index {sample_index} out of range [0, {len(df)})")
    row = df.iloc[sample_index]
    return {
        "sample_index": int(sample_index),
        "tool_index": int(row["TollIndex"]),
        "cycle": int(row["NumberOfCycle"]),
    }


def get_sample_sensor_data(sample_index: int):
    """
    Load a dataset row and extract the 120 sensor features.

    Returns
    -------
    dict[str, float]  — mapping of feature name → value
    """
    df = load_dataset()
    if sample_index < 0 or sample_index >= len(df):
        raise IndexError(f"Sample index {sample_index} out of range [0, {len(df)})")

    feature_names = get_feature_names()
    row = df.iloc[sample_index]
    return {f: float(row[f]) for f in feature_names}


# --------------------------------------------------
# PREDICTION WRAPPERS
# --------------------------------------------------

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
