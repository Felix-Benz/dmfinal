from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
from preprocess import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="readmitted", data=df)
    plt.title("Class Distribution of Readmission")
    plt.xlabel("Readmitted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    plot_class_distribution(df)
    plot_correlation_heatmap(df)

    print(f"EDA figures saved to: {FIGURES_DIR}")