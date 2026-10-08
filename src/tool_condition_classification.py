import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


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

    print(
        "\nDataset loaded:",
        df.shape
    )

    return df


# --------------------------------------------------
# FILTER TOOLS
# --------------------------------------------------

def filter_tools(df):

    tool_counts = (
        df.groupby("TollIndex")
        .size()
    )

    valid_tools = (
        tool_counts[
            tool_counts >= 10
        ]
        .index
    )

    removed_tools = (
        tool_counts[
            tool_counts < 10
        ]
        .index
        .tolist()
    )

    print(
        "\nRemoved tools:",
        removed_tools
    )

    df = df[
        df["TollIndex"].isin(
            valid_tools
        )
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
# CREATE TOOL CONDITION LABELS
# --------------------------------------------------

def create_condition_labels(df):

    def assign_condition(rul):

        if rul >= 0.75:
            return "Healthy"

        elif rul >= 0.50:
            return "Degrading"

        elif rul >= 0.25:
            return "Worn"

        else:
            return "Critical"

    df[
        "ToolCondition"
    ] = (
        df[
            "CycleToFailureNormalized"
        ]
        .apply(assign_condition)
    )

    print("\n")
    print("=" * 55)
    print("TOOL CONDITION DISTRIBUTION")
    print("=" * 55)

    condition_counts = (
        df[
            "ToolCondition"
        ]
        .value_counts()
    )

    print(
        condition_counts
    )

    print("\nPercentage distribution:")

    condition_percentage = (
        df[
            "ToolCondition"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    print(
        condition_percentage.round(2)
    )

    return df


# --------------------------------------------------
# SENSOR FEATURES
# --------------------------------------------------

def get_sensor_features(df):

    features = [
        column
        for column in df.columns
        if column.startswith(
            "Accelerometer"
        )
        or column.startswith(
            "Current"
        )
    ]

    print(
        "\nSensor features:",
        len(features)
    )

    return features


# --------------------------------------------------
# RANDOM FOREST CLASSIFIER
# --------------------------------------------------

def create_model():

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    return model


# --------------------------------------------------
# GROUP CROSS VALIDATION
# --------------------------------------------------

def cross_validate(
    df,
    features
):

    X = df[
        features
    ]

    y = df[
        "ToolCondition"
    ]

    groups = df[
        "TollIndex"
    ]

    group_kfold = GroupKFold(
        n_splits=5
    )

    fold_results = []

    all_true = []
    all_predicted = []

    print("\n")
    print("=" * 60)
    print("TOOL CONDITION GROUP CROSS VALIDATION")
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
            ]
            .unique()
        )

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        predictions = (
            model.predict(
                X_test
            )
        )

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )

        precision = (
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        recall = (
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        f1 = (
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        print(
            f"\nFold {fold_number}"
        )

        print(
            "Test tools:",
            test_tools
        )

        print(
            f"Accuracy       : {accuracy:.4f}"
        )

        print(
            f"Macro Precision: {precision:.4f}"
        )

        print(
            f"Macro Recall   : {recall:.4f}"
        )

        print(
            f"Macro F1       : {f1:.4f}"
        )

        fold_results.append({

            "Fold":
                fold_number,

            "TestTools":
                str(test_tools),

            "Accuracy":
                accuracy,

            "MacroPrecision":
                precision,

            "MacroRecall":
                recall,

            "MacroF1":
                f1
        })

        all_true.extend(
            y_test.tolist()
        )

        all_predicted.extend(
            predictions.tolist()
        )

        fold_number += 1

    return (
        pd.DataFrame(
            fold_results
        ),
        all_true,
        all_predicted
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def summarize_results(
    results_df
):

    print("\n")
    print("=" * 65)
    print("FINAL TOOL CONDITION CLASSIFICATION RESULT")
    print("=" * 65)

    average_accuracy = (
        results_df[
            "Accuracy"
        ]
        .mean()
    )

    average_precision = (
        results_df[
            "MacroPrecision"
        ]
        .mean()
    )

    average_recall = (
        results_df[
            "MacroRecall"
        ]
        .mean()
    )

    average_f1 = (
        results_df[
            "MacroF1"
        ]
        .mean()
    )

    accuracy_std = (
        results_df[
            "Accuracy"
        ]
        .std()
    )

    print(
        f"\nAverage Accuracy        : {average_accuracy:.4f}"
    )

    print(
        f"Average Macro Precision : {average_precision:.4f}"
    )

    print(
        f"Average Macro Recall    : {average_recall:.4f}"
    )

    print(
        f"Average Macro F1        : {average_f1:.4f}"
    )

    print(
        f"Accuracy Std Dev        : {accuracy_std:.4f}"
    )


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

def generate_classification_report(
    y_true,
    y_pred
):

    labels = [
        "Healthy",
        "Degrading",
        "Worn",
        "Critical"
    ]

    print("\n")
    print("=" * 65)
    print("OVERALL CLASSIFICATION REPORT")
    print("=" * 65)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    print(
        report
    )

    report_dict = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
    )

    report_df = (
        pd.DataFrame(
            report_dict
        )
        .transpose()
    )

    report_path = (
        RESULTS_DIR
        / "tool_condition_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        "Classification report saved:",
        report_path
    )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def create_confusion_matrix(
    y_true,
    y_pred
):

    labels = [
        "Healthy",
        "Degrading",
        "Worn",
        "Critical"
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display.plot(
        ax=ax,
        cmap=None,
        colorbar=False
    )

    plt.title(
        "Tool Condition Classification - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "tool_condition_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nConfusion matrix saved:",
        output_path
    )


# --------------------------------------------------
# TRAIN FINAL MODEL
# --------------------------------------------------

def train_final_model(
    df,
    features
):

    import joblib

    X = df[
        features
    ]

    y = df[
        "ToolCondition"
    ]

    model = create_model()

    model.fit(
        X,
        y
    )

    model_path = (
        MODELS_DIR
        / "tool_condition_random_forest.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nFinal classification model saved:",
        model_path
    )


# --------------------------------------------------
# SAVE CROSS-VALIDATION RESULTS
# --------------------------------------------------

def save_results(
    results_df
):

    result_path = (
        RESULTS_DIR
        / "tool_condition_cross_validation.csv"
    )

    results_df.to_csv(
        result_path,
        index=False
    )

    print(
        "Cross-validation results saved:",
        result_path
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    df = filter_tools(
        df
    )

    df = create_condition_labels(
        df
    )

    features = get_sensor_features(
        df
    )

    (
        results_df,
        y_true,
        y_pred
    ) = cross_validate(
        df,
        features
    )

    summarize_results(
        results_df
    )

    generate_classification_report(
        y_true,
        y_pred
    )

    create_confusion_matrix(
        y_true,
        y_pred
    )

    save_results(
        results_df
    )

    train_final_model(
        df,
        features
    )

    print("\n")
    print("=" * 65)
    print("TOOL CONDITION CLASSIFICATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()