import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "FeatureAndMetadata_Milling.csv"
)

RESULTS_DIR = BASE_DIR / "outputs" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
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

    print("\nOriginal dataset:", df.shape)

    return df


# --------------------------------------------------
# REMOVE TOOLS WITHOUT LIFECYCLE INFORMATION
# --------------------------------------------------

def remove_insufficient_tools(df):

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
        "\nTools removed because they have "
        "fewer than 10 cycles:",
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
# GROUP CROSS VALIDATION
# --------------------------------------------------

def run_cross_validation(
    df,
    features
):

    X = df[features]

    y = df[
        "CycleToFailureNormalized"
    ]

    groups = df[
        "TollIndex"
    ]

    # 5-fold tool-based validation
    group_kfold = GroupKFold(
        n_splits=5
    )

    results = []

    print("\n==============================")
    print("GROUP CROSS VALIDATION")
    print("==============================")

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

        train_tools = sorted(
            groups.iloc[
                train_index
            ].unique()
        )

        test_tools = sorted(
            groups.iloc[
                test_index
            ].unique()
        )

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
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

        results.append({
            "Fold": fold_number,
            "TestTools":
                str(test_tools),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        fold_number += 1

    return pd.DataFrame(results)


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

def show_summary(results):

    print("\n==============================")
    print("FINAL CROSS-VALIDATION RESULT")
    print("==============================")

    print(
        "\nAverage MAE :",
        round(
            results["MAE"].mean(),
            4
        )
    )

    print(
        "Average RMSE:",
        round(
            results["RMSE"].mean(),
            4
        )
    )

    print(
        "Average R²  :",
        round(
            results["R2"].mean(),
            4
        )
    )

    print(
        "\nR² Standard Deviation:",
        round(
            results["R2"].std(),
            4
        )
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    path = (
        RESULTS_DIR
        / "rul_group_cross_validation.csv"
    )

    results.to_csv(
        path,
        index=False
    )

    print(
        "\nResults saved:",
        path
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    df = remove_insufficient_tools(
        df
    )

    features = get_sensor_features(
        df
    )

    results = run_cross_validation(
        df,
        features
    )

    show_summary(results)

    save_results(results)

    print(
        "\nRUL validation completed."
    )


if __name__ == "__main__":
    main()