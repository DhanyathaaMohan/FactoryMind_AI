import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        header=1
    )

    print("\nDataset loaded successfully")
    print("Shape:", df.shape)

    return df


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

def show_basic_information(df):

    print("\n============================")
    print("DATASET INFORMATION")
    print("============================")

    print("\nNumber of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nData types:")
    print(df.dtypes.value_counts())


# --------------------------------------------------
# TOOL INFORMATION
# --------------------------------------------------

def analyze_tools(df):

    print("\n============================")
    print("TOOL INFORMATION")
    print("============================")

    tools = sorted(df["TollIndex"].unique())

    print("\nNumber of tools:", len(tools))
    print("Tool IDs:", tools)

    print("\nCycles available for each tool:")

    cycle_counts = (
        df.groupby("TollIndex")
        .size()
        .sort_index()
    )

    print(cycle_counts)


# --------------------------------------------------
# RUL INFORMATION
# --------------------------------------------------

def analyze_rul(df):

    print("\n============================")
    print("RUL INFORMATION")
    print("============================")

    print(
        df[
            [
                "TollIndex",
                "NumberOfCycle",
                "CycleToFailure",
                "CycleToFailureNormalized"
            ]
        ].head(20)
    )

    print("\nCycleToFailure statistics:")

    print(
        df["CycleToFailure"].describe()
    )


# --------------------------------------------------
# CONVERT NORMALIZED RUL
# --------------------------------------------------

def clean_normalized_rul(df):

    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    return df


# --------------------------------------------------
# VISUALIZE TOOL LIFECYCLE
# --------------------------------------------------

def plot_tool_lifecycle(df):

    tool_ids = sorted(df["TollIndex"].unique())

    for tool_id in tool_ids:

        tool_data = df[
            df["TollIndex"] == tool_id
        ].sort_values("NumberOfCycle")

        plt.figure(figsize=(8, 5))

        plt.plot(
            tool_data["NumberOfCycle"],
            tool_data["CycleToFailure"]
        )

        plt.xlabel("Milling Cycle")
        plt.ylabel("Cycles Remaining")

        plt.title(
            f"Tool {tool_id} - Remaining Useful Life"
        )

        plt.grid(True)

        save_path = (
            OUTPUT_DIR /
            f"tool_{tool_id}_rul.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

    print(
        "\nRUL lifecycle plots saved inside:",
        OUTPUT_DIR
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_dataset()

    show_basic_information(df)

    analyze_tools(df)

    df = clean_normalized_rul(df)

    analyze_rul(df)

    plot_tool_lifecycle(df)

    print("\nDataset exploration completed successfully.")


if __name__ == "__main__":
    main()