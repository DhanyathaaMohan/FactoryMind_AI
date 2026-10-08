import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# FILTER TOOLS
# --------------------------------------------------

def filter_tools(df):

    tool_counts = (
        df.groupby("TollIndex")
        .size()
    )

    valid_tools = tool_counts[
        tool_counts >= 10
    ].index

    removed_tools = tool_counts[
        tool_counts < 10
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

    sensor_features = [
        column
        for column in df.columns
        if column.startswith("Accelerometer")
        or column.startswith("Current")
    ]

    print(
        "\nTotal sensor features:",
        len(sensor_features)
    )

    return sensor_features


# --------------------------------------------------
# FEATURE RANKING
# --------------------------------------------------

def rank_features(df, sensor_features):

    correlations = []

    for feature in sensor_features:

        correlation = (
            df[feature]
            .corr(
                df["CycleToFailureNormalized"],
                method="spearman"
            )
        )

        correlations.append({
            "Feature": feature,
            "Correlation": correlation,
            "AbsoluteCorrelation": abs(correlation)
        })

    correlation_df = pd.DataFrame(
        correlations
    )

    correlation_df = (
        correlation_df
        .sort_values(
            "AbsoluteCorrelation",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop 20 ranked features:")
    print(
        correlation_df[
            [
                "Feature",
                "Correlation",
                "AbsoluteCorrelation"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    correlation_path = (
        RESULTS_DIR
        / "rul_feature_ranking.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    print(
        "\nFeature ranking saved:",
        correlation_path
    )

    return correlation_df


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

def create_model():

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    return model


# --------------------------------------------------
# EVALUATE FEATURE SUBSET
# --------------------------------------------------

def evaluate_feature_subset(
    df,
    selected_features,
    subset_name
):

    X = df[selected_features]

    y = df[
        "CycleToFailureNormalized"
    ]

    groups = df[
        "TollIndex"
    ]

    group_kfold = GroupKFold(
        n_splits=5
    )

    fold_results = []

    print("\n")
    print("=" * 60)
    print(
        f"FEATURE SET: {subset_name}"
    )
    print(
        f"Number of features: {len(selected_features)}"
    )
    print("=" * 60)

    fold_number = 1

    for train_index, test_index in group_kfold.split(
        X,
        y,
        groups
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        test_tools = sorted(
            groups.iloc[
                test_index
            ].unique()
        )

        model = create_model()

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

        fold_results.append({
            "FeatureSet": subset_name,
            "NumberOfFeatures":
                len(selected_features),
            "Fold": fold_number,
            "TestTools":
                str(test_tools),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        fold_number += 1

    return fold_results


# --------------------------------------------------
# FEATURE SUBSET COMPARISON
# --------------------------------------------------

def compare_feature_subsets(
    df,
    sensor_features,
    ranked_features
):

    feature_sets = {

        "All 120 Features":
            sensor_features,

        "Top 60 Features":
            ranked_features[
                "Feature"
            ]
            .head(60)
            .tolist(),

        "Top 30 Features":
            ranked_features[
                "Feature"
            ]
            .head(30)
            .tolist(),

        "Top 20 Features":
            ranked_features[
                "Feature"
            ]
            .head(20)
            .tolist(),

        "Top 10 Features":
            ranked_features[
                "Feature"
            ]
            .head(10)
            .tolist()
    }

    all_results = []

    for subset_name, features in feature_sets.items():

        subset_results = (
            evaluate_feature_subset(
                df,
                features,
                subset_name
            )
        )

        all_results.extend(
            subset_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    return results_df


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def summarize_results(results_df):

    summary = (
        results_df
        .groupby(
            [
                "FeatureSet",
                "NumberOfFeatures"
            ]
        )
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
    print("=" * 75)
    print("FINAL FEATURE SELECTION COMPARISON")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# --------------------------------------------------
# SAVE BEST FEATURE SET
# --------------------------------------------------

def save_best_features(
    summary,
    ranked_features
):

    best_row = summary.iloc[0]

    best_count = int(
        best_row[
            "NumberOfFeatures"
        ]
    )

    best_name = best_row[
        "FeatureSet"
    ]

    if best_count == 120:

        best_features = (
            ranked_features[
                "Feature"
            ]
            .tolist()
        )

    else:

        best_features = (
            ranked_features[
                "Feature"
            ]
            .head(best_count)
            .tolist()
        )

    best_feature_df = pd.DataFrame({
        "Feature": best_features
    })

    best_path = (
        RESULTS_DIR
        / "best_rul_features.csv"
    )

    best_feature_df.to_csv(
        best_path,
        index=False
    )

    print(
        "\nBest feature set:",
        best_name
    )

    print(
        "Number of selected features:",
        best_count
    )

    print(
        "Best Average R²:",
        round(
            best_row[
                "Average_R2"
            ],
            4
        )
    )

    print(
        "Best features saved:",
        best_path
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(
    results_df,
    summary
):

    detailed_path = (
        RESULTS_DIR
        / "rul_feature_selection_folds.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "rul_feature_selection_summary.csv"
    )

    results_df.to_csv(
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

    sensor_features = (
        get_sensor_features(
            df
        )
    )

    ranked_features = (
        rank_features(
            df,
            sensor_features
        )
    )

    results_df = (
        compare_feature_subsets(
            df,
            sensor_features,
            ranked_features
        )
    )

    summary = (
        summarize_results(
            results_df
        )
    )

    save_results(
        results_df,
        summary
    )

    save_best_features(
        summary,
        ranked_features
    )

    print(
        "\nFeature selection completed successfully."
    )


if __name__ == "__main__":
    main()