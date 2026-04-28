import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
from preprocess import clean_data
from config import FIGURES_DIR, TARGET_COLUMN


def run_eda():
    """
    Create basic exploratory data analysis plots.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = clean_data(df)

    print("\nDataset shape after cleaning:")
    print(df.shape)

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    # Class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COLUMN, data=df)
    plt.title("30-Day Readmission Class Distribution")
    plt.xlabel("Readmitted Within 30 Days")
    plt.ylabel("Count")
    plt.savefig(FIGURES_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close()

    # Time in hospital histogram
    if "time_in_hospital" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df["time_in_hospital"], bins=14, kde=False)
        plt.title("Distribution of Time in Hospital")
        plt.xlabel("Days in Hospital")
        plt.ylabel("Count")
        plt.savefig(FIGURES_DIR / "time_in_hospital_distribution.png", bbox_inches="tight")
        plt.close()

    # Numerical correlation heatmap
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap of Numerical Features")
        plt.savefig(FIGURES_DIR / "correlation_heatmap.png", bbox_inches="tight")
        plt.close()

    print("\nEDA plots saved to outputs/figures/")


if __name__ == "__main__":
    run_eda()