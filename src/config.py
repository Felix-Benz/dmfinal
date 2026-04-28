from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetic_data.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_diabetes_data.csv"

# Output paths
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODELS_DIR = ROOT_DIR / "outputs" / "models"

# Target column
TARGET_COLUMN = "readmitted"

# Random seed for reproducibility
RANDOM_STATE = 42