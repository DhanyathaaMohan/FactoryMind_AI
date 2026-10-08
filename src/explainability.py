"""
explainability.py
-----------------
Wraps the final Random Forest model with a SHAP TreeExplainer and
exposes a function that explains a single sensor-data sample.

Each explanation entry reports:
    feature        – sensor feature name
    feature_value  – raw sensor reading
    shap_value     – SHAP contribution to this prediction
    direction      – human-readable: "pushes RUL higher" or "pushes RUL lower"

The explainer is cached at module level so it is only built once per
process (TreeExplainer construction is the expensive step).
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict, Any

# Reuse the model loader from predictive_maintenance
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictive_maintenance import load_final_model


# --------------------------------------------------
# MODULE-LEVEL CACHE
# --------------------------------------------------

_model = None
_features = None
_explainer = None


def _get_explainer():
    """
    Lazily initialise and cache the TreeExplainer.
    Re-uses the same model/features loaded by predictive_maintenance.
    SHAP is imported here (not at module level) to avoid loading
    numba/llvmlite DLLs on import — those may be blocked by
    Application Control policies on some Windows environments.
    """
    global _model, _features, _explainer

    if _explainer is None:
        import shap  # lazy import — avoids numba DLL load at startup
        print("Loading final model for SHAP explainer …")
        _model, _features = load_final_model()
        _explainer = shap.TreeExplainer(_model)
        print("SHAP TreeExplainer ready.")

    return _explainer, _model, _features


# --------------------------------------------------
# SINGLE-SAMPLE EXPLANATION
# --------------------------------------------------

def explain_sample(
    sensor_data: dict,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Compute SHAP values for a single sensor reading and return the
    top-N most influential features.

    Parameters
    ----------
    sensor_data : dict
        Mapping of feature_name -> float.  Must include all 120
        sensor features (extra keys ignored).
    top_n : int
        Number of top features to return (default 10; pass 5 for a
        compact summary).

    Returns
    -------
    list of dicts, sorted by |shap_value| descending:
        {
            "feature":       str,
            "feature_value": float,
            "shap_value":    float,
            "direction":     str   "pushes RUL higher" | "pushes RUL lower"
        }
    """
    explainer, model, features = _get_explainer()

    # Missing-feature guard
    missing = [f for f in features if f not in sensor_data]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} sensor feature(s): {missing[:5]} …"
        )

    # Build input row in correct column order
    row = pd.DataFrame([{f: sensor_data[f] for f in features}])

    # Compute SHAP values — returns shape (1, n_features) for regressors
    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        # Some older SHAP versions wrap in a list for single-output regressors
        shap_values = shap_values[0]

    shap_values = np.array(shap_values).flatten()
    feature_values = row.iloc[0].values

    # Build sorted result list
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    results = []
    for idx in top_indices:
        sv = float(shap_values[idx])
        results.append({
            "feature":       features[idx],
            "feature_value": float(feature_values[idx]),
            "shap_value":    sv,
            "direction":     "pushes RUL higher" if sv > 0 else "pushes RUL lower",
        })

    return results


# --------------------------------------------------
# MAIN — demonstration
# --------------------------------------------------

def main():
    import json
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    import pandas as pd

    DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"

    df = pd.read_csv(DATA_PATH, sep=";", header=1)
    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    _, features = load_final_model()
    sample = df[features].iloc[0].to_dict()

    print("\n" + "=" * 65)
    print("SHAP LOCAL EXPLANATION (sample 0)")
    print("=" * 65)

    results = explain_sample(sample, top_n=10)

    for rank, item in enumerate(results, start=1):
        print(
            f"  [{rank:02d}] {item['feature']:<55} "
            f"SHAP={item['shap_value']:+.4f}  "
            f"({item['direction']})"
        )


if __name__ == "__main__":
    main()
