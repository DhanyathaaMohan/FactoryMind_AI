import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "FeatureAndMetadata_Milling.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "outputs"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        header=1
    )

    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    print("\nDataset loaded:", df.shape)

    return df


# --------------------------------------------------
# REMOVE TOOLS WITH INSUFFICIENT LIFECYCLE
# --------------------------------------------------

def filter_tools(df):

    counts = (
        df.groupby("TollIndex")
        .size()
    )

    valid_tools = counts[
        counts >= 10
    ].index

    removed_tools = counts[
        counts < 10
    ].index.tolist()

    print(
        "\nRemoved tools:",
        removed_tools
    )

    df = df[
        df["TollIndex"].isin(valid_tools)
    ].copy()

    print(
        "Dataset after filtering:",
        df.shape
    )

    print(
        "Remaining tools:",
        df["TollIndex"].nunique()
    )

    return df


# --------------------------------------------------
# SENSOR FEATURES
# --------------------------------------------------

def get_sensor_features(df):

    features = [
        column
        for column in df.columns
        if column.startswith("Accelerometer")
        or column.startswith("Current")
    ]

    print(
        "\nSensor features:",
        len(features)
    )

    return features


# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------

def get_models():

    models = {

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
    }

    return models


# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

def evaluate_models(
    df,
    features,
    models
):

    X = df[features]

    y = df[
        "CycleToFailureNormalized"
    ]

    groups = df[
        "TollIndex"
    ]

    group_kfold = GroupKFold(
        n_splits=5
    )

    all_results = []

    for model_name, model in models.items():

        print("\n")
        print("=" * 45)
        print(model_name.upper())
        print("=" * 45)

        fold_number = 1

        for train_index, test_index in group_kfold.split(
            X,
            y,
            groups
        ):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            test_tools = sorted(
                groups.iloc[
                    test_index
                ].unique()
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            print(
                f"\nFold {fold_number}"
            )

            print(
                "Test tools:",
                test_tools
            )

            print(
                f"MAE  : {mae:.4f}"
            )

            print(
                f"RMSE : {rmse:.4f}"
            )

            print(
                f"R²   : {r2:.4f}"
            )

            all_results.append({
                "Model": model_name,
                "Fold": fold_number,
                "TestTools":
                    str(test_tools),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            fold_number += 1

    return pd.DataFrame(
        all_results
    )


# --------------------------------------------------
# MODEL SUMMARY
# --------------------------------------------------

def summarize_results(results):

    summary = (
        results
        .groupby("Model")
        .agg(
            Average_MAE=("MAE", "mean"),
            Average_RMSE=("RMSE", "mean"),
            Average_R2=("R2", "mean"),
            R2_Std=("R2", "std")
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "Average_R2",
        ascending=False
    )

    print("\n")
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(
    results,
    summary
):

    detailed_path = (
        RESULTS_DIR
        / "rul_model_comparison_folds.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "rul_model_comparison_summary.csv"
    )

    results.to_csv(
        detailed_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        "\nDetailed results saved:",
        detailed_path
    )

    print(
        "Summary saved:",
        summary_path
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    df = filter_tools(
        df
    )

    features = get_sensor_features(
        df
    )

    models = get_models()

    results = evaluate_models(
        df,
        features,
        models
    )

    summary = summarize_results(
        results
    )

    save_results(
        results,
        summary
    )

    print(
        "\nModel comparison completed successfully."
    )


if __name__ == "__main__":
    main()