import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"

RESULTS_DIR = BASE_DIR / "outputs" / "results"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

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

    # Convert normalized RUL from comma decimal format
    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    print("\nDataset loaded:", df.shape)

    return df


# --------------------------------------------------
# IDENTIFY MACHINE FEATURES
# --------------------------------------------------

def identify_sensor_features(df):

    # Columns that should NOT be treated as sensor features
    metadata_columns = [
        "NumberOfCycle",
        "SampleIndex",
        "TollIndex",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength",
        "CycleToFailure",
        "CycleToFailureNormalized"
    ]

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    sensor_features = [
        column
        for column in numeric_columns
        if column not in metadata_columns
    ]

    print("\n==============================")
    print("FEATURE INFORMATION")
    print("==============================")

    print("\nNumeric columns:", len(numeric_columns))
    print("Machine/Sensor features:", len(sensor_features))

    print("\nFirst 10 sensor features:")

    for feature in sensor_features[:10]:
        print(feature)

    return sensor_features


# --------------------------------------------------
# FEATURE CORRELATION WITH RUL
# --------------------------------------------------

def calculate_rul_correlation(df, sensor_features):

    target = "CycleToFailureNormalized"

    correlations = {}

    for feature in sensor_features:

        correlation = df[feature].corr(
            df[target],
            method="spearman"
        )

        correlations[feature] = correlation

    correlation_df = pd.DataFrame({
        "Feature": correlations.keys(),
        "Correlation": correlations.values()
    })

    correlation_df["AbsoluteCorrelation"] = (
        correlation_df["Correlation"].abs()
    )

    correlation_df = correlation_df.sort_values(
        "AbsoluteCorrelation",
        ascending=False
    )

    return correlation_df


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_feature_ranking(correlation_df):

    output_path = (
        RESULTS_DIR /
        "feature_rul_correlation.csv"
    )

    correlation_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nFeature correlation ranking saved to:"
    )

    print(output_path)


# --------------------------------------------------
# DISPLAY TOP FEATURES
# --------------------------------------------------

def show_top_features(correlation_df):

    print("\n==============================")
    print("TOP FEATURES RELATED TO RUL")
    print("==============================")

    print(
        correlation_df[
            [
                "Feature",
                "Correlation",
                "AbsoluteCorrelation"
            ]
        ].head(20).to_string(index=False)
    )


# --------------------------------------------------
# PLOT TOP FEATURES
# --------------------------------------------------

def plot_top_features(correlation_df):

    top_features = correlation_df.head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"],
        top_features["AbsoluteCorrelation"]
    )

    plt.xlabel(
        "Absolute Spearman Correlation with RUL"
    )

    plt.ylabel(
        "Machine Feature"
    )

    plt.title(
        "Top 20 Features Related to Tool Remaining Useful Life"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "top_20_rul_features.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nFeature importance plot saved to:"
    )

    print(output_path)


# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

def feature_statistics(df, sensor_features):

    statistics = (
        df[sensor_features]
        .describe()
        .transpose()
    )

    output_path = (
        RESULTS_DIR /
        "sensor_feature_statistics.csv"
    )

    statistics.to_csv(output_path)

    print(
        "\nSensor feature statistics saved to:"
    )

    print(output_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    sensor_features = identify_sensor_features(df)

    correlation_df = calculate_rul_correlation(
        df,
        sensor_features
    )

    show_top_features(correlation_df)

    save_feature_ranking(correlation_df)

    feature_statistics(
        df,
        sensor_features
    )

    plot_top_features(correlation_df)

    print("\nFeature analysis completed successfully.")


if __name__ == "__main__":
    main()