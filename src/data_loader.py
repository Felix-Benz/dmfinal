import pandas as pd
from config import RAW_DATA_PATH


def load_data(path=RAW_DATA_PATH):
    """
    Load the raw UCI Diabetes hospital readmission dataset.

    The dataset should be stored at:
    data/raw/diabetic_data.csv
    """
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place diabetic_data.csv inside data/raw/ before running."
        )


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.info())