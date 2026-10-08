import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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
# SELECT SENSOR FEATURES
# --------------------------------------------------

def select_sensor_features(df):

    sensor_features = [
        column
        for column in df.columns
        if column.startswith("Accelerometer")
        or column.startswith("Current")
    ]

    accelerometer_features = [
        x for x in sensor_features
        if x.startswith("Accelerometer")
    ]

    current_features = [
        x for x in sensor_features
        if x.startswith("Current")
    ]

    print("\n==============================")
    print("MODEL FEATURES")
    print("==============================")

    print("\nTotal sensor features:", len(sensor_features))
    print("Accelerometer features:", len(accelerometer_features))
    print("Current features:", len(current_features))

    return sensor_features


# --------------------------------------------------
# PREPARE INPUT AND TARGET
# --------------------------------------------------

def prepare_data(df, sensor_features):

    X = df[sensor_features]

    y = df["CycleToFailureNormalized"]

    groups = df["TollIndex"]

    return X, y, groups


# --------------------------------------------------
# GROUP-BASED SPLIT
# --------------------------------------------------

def split_dataset(X, y, groups):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    train_tools = groups.iloc[train_index]
    test_tools = groups.iloc[test_index]

    print("\n==============================")
    print("GROUP-BASED DATA SPLIT")
    print("==============================")

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print(
        "\nTraining tools:",
        sorted(train_tools.unique())
    )

    print(
        "Testing tools:",
        sorted(test_tools.unique())
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_tools
    )


# --------------------------------------------------
# TRAIN RANDOM FOREST
# --------------------------------------------------

def train_model(X_train, y_train):

    print("\nTraining Random Forest RUL model...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    return model


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

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

    print("\n==============================")
    print("RUL MODEL PERFORMANCE")
    print("==============================")

    print(f"\nMAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return predictions


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

def save_predictions(
    y_test,
    predictions,
    test_tools
):

    results = pd.DataFrame({
        "ToolID": test_tools.values,
        "Actual_RUL": y_test.values,
        "Predicted_RUL": predictions
    })

    results["Absolute_Error"] = (
        results["Actual_RUL"]
        - results["Predicted_RUL"]
    ).abs()

    path = RESULTS_DIR / "rul_predictions.csv"

    results.to_csv(
        path,
        index=False
    )

    print("\nPredictions saved:", path)

    return results


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

def feature_importance(
    model,
    sensor_features
):

    importance = pd.DataFrame({
        "Feature": sensor_features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n==============================")
    print("TOP 15 MODEL FEATURES")
    print("==============================")

    print(
        importance.head(15).to_string(
            index=False
        )
    )

    path = (
        RESULTS_DIR /
        "rul_feature_importance.csv"
    )

    importance.to_csv(
        path,
        index=False
    )


# --------------------------------------------------
# PLOT RESULTS
# --------------------------------------------------

def plot_results(results):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        results["Actual_RUL"],
        results["Predicted_RUL"],
        alpha=0.6
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Normalized RUL"
    )

    plt.ylabel(
        "Predicted Normalized RUL"
    )

    plt.title(
        "Actual vs Predicted RUL"
    )

    plt.grid(True)

    plt.tight_layout()

    path = (
        PLOTS_DIR /
        "actual_vs_predicted_rul.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print("\nRUL prediction plot saved:", path)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

def save_model(model):

    path = (
        MODEL_DIR /
        "rul_random_forest.pkl"
    )

    joblib.dump(
        model,
        path
    )

    print("\nModel saved:", path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    sensor_features = select_sensor_features(df)

    X, y, groups = prepare_data(
        df,
        sensor_features
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_tools
    ) = split_dataset(
        X,
        y,
        groups
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    results = save_predictions(
        y_test,
        predictions,
        test_tools
    )

    feature_importance(
        model,
        sensor_features
    )

    plot_results(results)

    save_model(model)

    print(
        "\nFactoryMind RUL baseline completed successfully."
    )


if __name__ == "__main__":
    main()