"""
predictive_maintenance.py
--------------------------
Loads the dataset, trains the final Random Forest RUL model on all
available data, saves the model and the ordered feature list, and
exposes a reusable inference function used by the backend and the
maintenance assistant.

Saved artefacts
---------------
models/rul_random_forest_final.pkl  – trained RandomForestRegressor
models/rul_feature_names.json       – ordered list of 120 sensor features
"""

import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "rul_random_forest_final.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "rul_feature_names.json"


# --------------------------------------------------
# TOOL CONDITION MAPPING
# --------------------------------------------------

def map_tool_condition(rul: float) -> str:
    """
    Derive categorical tool condition from a predicted
    normalised RUL value in [0, 1].

    Thresholds
    ----------
    >= 0.75        -> Healthy
    0.50 – 0.74    -> Degrading
    0.25 – 0.49    -> Worn
    < 0.25         -> Critical
    """
    if rul >= 0.75:
        return "Healthy"
    elif rul >= 0.50:
        return "Degrading"
    elif rul >= 0.25:
        return "Worn"
    else:
        return "Critical"


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """
    Load the milling CSV.  The file uses a semicolon separator and
    has a second header row (header=1).  The target column uses a
    comma as the decimal separator in some locales so we normalise
    it to a float.
    """
    df = pd.read_csv(DATA_PATH, sep=";", header=1)

    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

def get_sensor_features(df: pd.DataFrame) -> list:
    """
    Return the ordered list of all Accelerometer and Current sensor
    columns.  Lifecycle columns and metadata columns are excluded.

    Excluded from model input:
        NumberOfCycle, CycleToFailure,
        CycleToFailureNormalized, TollIndex
    """
    features = [
        col for col in df.columns
        if col.startswith("Accelerometer") or col.startswith("Current")
    ]
    print(f"Sensor features: {len(features)}")
    return features


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    features: list
) -> RandomForestRegressor:
    """
    Train the Random Forest on all available data (no hold-out split).
    This is the production model — cross-validation results are already
    documented in outputs/results/.
    """
    X = df[features]
    y = df["CycleToFailureNormalized"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print(
        f"\nTraining final Random Forest on {len(X)} samples "
        f"with {len(features)} features …"
    )
    model.fit(X, y)
    print("Training complete.")
    return model


# --------------------------------------------------
# PERSISTENCE
# --------------------------------------------------

def save_model(model: RandomForestRegressor, features: list) -> None:
    """Persist the model and its feature list to disk."""
    joblib.dump(model, FINAL_MODEL_PATH)
    print(f"Final model saved: {FINAL_MODEL_PATH}")

    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"Feature names saved: {FEATURE_NAMES_PATH}")


def load_final_model():
    """
    Load the persisted model and feature list.

    Returns
    -------
    model    : RandomForestRegressor
    features : list[str]
    """
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {FINAL_MODEL_PATH}. "
            "Run predictive_maintenance.py first."
        )
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names not found at {FEATURE_NAMES_PATH}. "
            "Run predictive_maintenance.py first."
        )

    model = joblib.load(FINAL_MODEL_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    return model, features


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

def predict(sensor_data: dict) -> dict:
    """
    Run a single-sample RUL prediction.

    Parameters
    ----------
    sensor_data : dict
        Mapping of feature_name -> float value.  Must contain all
        120 sensor features.  Extra keys are silently ignored; missing
        keys raise a ValueError.

    Returns
    -------
    dict with keys:
        predicted_rul  : float   clamped to [0, 1]
        tool_condition : str     derived from RUL threshold map
    """
    model, features = load_final_model()

    # Validate that every required feature is present
    missing = [f for f in features if f not in sensor_data]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required sensor feature(s): "
            f"{missing[:5]} …"
        )

    # Build input in the exact column order the model was trained on
    row = pd.DataFrame([{f: sensor_data[f] for f in features}])

    raw_rul = float(model.predict(row)[0])

    # Clamp to [0, 1]
    predicted_rul = float(np.clip(raw_rul, 0.0, 1.0))

    tool_condition = map_tool_condition(predicted_rul)

    return {
        "predicted_rul": predicted_rul,
        "tool_condition": tool_condition,
    }


# --------------------------------------------------
# MAIN — train and save
# --------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("FACTORYMIND — FINAL RUL MODEL TRAINING")
    print("=" * 65)

    df = load_dataset()
    features = get_sensor_features(df)
    model = train_final_model(df, features)
    save_model(model, features)

    # Quick sanity check on a random sample
    sample_row = df[features].iloc[0].to_dict()
    result = predict(sample_row)

    print("\n--- Inference sanity check ---")
    print(f"  Predicted RUL  : {result['predicted_rul']:.4f}")
    print(f"  Tool Condition : {result['tool_condition']}")
    print("\nFinal model training completed successfully.")


if __name__ == "__main__":
    main()
