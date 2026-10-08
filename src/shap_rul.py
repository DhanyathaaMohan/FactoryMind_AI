import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


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

PLOTS_DIR = (
    BASE_DIR
    / "outputs"
    / "plots"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
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
# GET SENSOR FEATURES
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
# TRAIN FINAL RANDOM FOREST
# --------------------------------------------------

def train_model(df, features):

    X = df[features]

    y = df[
        "CycleToFailureNormalized"
    ]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print(
        "\nTraining Random Forest using all 120 sensor features..."
    )

    model.fit(
        X,
        y
    )

    print(
        "Model training completed."
    )

    return model, X, y


# --------------------------------------------------
# SHAP ANALYSIS
# --------------------------------------------------

def calculate_shap(model, X):

    print(
        "\nCalculating SHAP values..."
    )

    # Use maximum 300 samples for faster SHAP analysis
    if len(X) > 300:

        X_shap = X.sample(
            n=300,
            random_state=42
        )

    else:

        X_shap = X.copy()

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_shap
    )

    shap_values = np.array(
        shap_values
    )

    print(
        "SHAP samples:",
        len(X_shap)
    )

    print(
        "SHAP calculation completed."
    )

    return (
        explainer,
        shap_values,
        X_shap
    )


# --------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

def calculate_global_importance(
    shap_values,
    X_shap
):

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "Feature": X_shap.columns,
        "MeanAbsoluteSHAP":
            mean_abs_shap
    })

    importance_df = (
        importance_df
        .sort_values(
            "MeanAbsoluteSHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 75)
    print("TOP 20 SHAP FEATURES")
    print("=" * 75)

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    output_path = (
        RESULTS_DIR
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSHAP feature importance saved:",
        output_path
    )

    return importance_df


# --------------------------------------------------
# SHAP BAR PLOT
# --------------------------------------------------

def create_bar_plot(
    shap_values,
    X_shap
):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "shap_rul_feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP bar plot saved:",
        output_path
    )


# --------------------------------------------------
# SHAP BEESWARM PLOT
# --------------------------------------------------

def create_summary_plot(
    shap_values,
    X_shap
):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_shap,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "shap_rul_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP summary plot saved:",
        output_path
    )


# --------------------------------------------------
# TOP FEATURE DEPENDENCE PLOT
# --------------------------------------------------

def create_dependence_plot(
    shap_values,
    X_shap,
    importance_df
):

    top_feature = (
        importance_df
        .iloc[0]["Feature"]
    )

    print(
        "\nCreating dependence plot for:",
        top_feature
    )

    plt.figure()

    shap.dependence_plot(
        top_feature,
        shap_values,
        X_shap,
        interaction_index=None,
        show=False
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "shap_top_feature_dependence.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Dependence plot saved:",
        output_path
    )


# --------------------------------------------------
# LOCAL EXPLANATION
# --------------------------------------------------

def create_local_explanation(
    explainer,
    shap_values,
    X_shap,
    model
):

    sample_index = 0

    sample = X_shap.iloc[
        sample_index:
        sample_index + 1
    ]

    prediction = model.predict(
        sample
    )[0]

    print("\n")
    print("=" * 75)
    print("LOCAL SHAP EXPLANATION")
    print("=" * 75)

    print(
        "Predicted normalized RUL:",
        round(
            prediction,
            4
        )
    )

    local_values = shap_values[
        sample_index
    ]

    local_df = pd.DataFrame({
        "Feature":
            X_shap.columns,
        "FeatureValue":
            sample.iloc[0].values,
        "SHAPValue":
            local_values
    })

    local_df[
        "AbsoluteSHAP"
    ] = np.abs(
        local_df[
            "SHAPValue"
        ]
    )

    local_df = (
        local_df
        .sort_values(
            "AbsoluteSHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop 10 factors affecting this prediction:"
    )

    print(
        local_df[
            [
                "Feature",
                "FeatureValue",
                "SHAPValue"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    output_path = (
        RESULTS_DIR
        / "shap_local_explanation.csv"
    )

    local_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nLocal explanation saved:",
        output_path
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

    model, X, y = train_model(
        df,
        features
    )

    (
        explainer,
        shap_values,
        X_shap
    ) = calculate_shap(
        model,
        X
    )

    importance_df = (
        calculate_global_importance(
            shap_values,
            X_shap
        )
    )

    create_bar_plot(
        shap_values,
        X_shap
    )

    create_summary_plot(
        shap_values,
        X_shap
    )

    create_dependence_plot(
        shap_values,
        X_shap,
        importance_df
    )

    create_local_explanation(
        explainer,
        shap_values,
        X_shap,
        model
    )

    print("\n")
    print("=" * 75)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 75)

    print(
        "\nGenerated files:"
    )

    print(
        "1. shap_feature_importance.csv"
    )

    print(
        "2. shap_local_explanation.csv"
    )

    print(
        "3. shap_rul_feature_importance.png"
    )

    print(
        "4. shap_rul_summary.png"
    )

    print(
        "5. shap_top_feature_dependence.png"
    )


if __name__ == "__main__":
    main()