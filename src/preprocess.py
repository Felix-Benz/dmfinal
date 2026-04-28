import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import TARGET_COLUMN, RANDOM_STATE


def clean_data(df):
    """
    Basic cleaning for the UCI Diabetes dataset.

    Main changes:
    - Replace '?' with missing values
    - Create a binary target for 30-day readmission
    - Drop ID-style columns that do not help generalize
    """

    df = df.copy()

    # Replace invalid missing value marker
    df = df.replace("?", np.nan)

    # Binary target:
    # '<30' means readmitted within 30 days
    # '>30' and 'NO' are treated as not readmitted within 30 days
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == "<30" else 0)

    # Drop columns that are identifiers or mostly not useful for prediction
    columns_to_drop = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
    ]

    existing_drop_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    return df


def split_features_target(df):
    """
    Split dataframe into X features and y target.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor(X):
    """
    Build preprocessing pipeline for numerical and categorical columns.

    Numerical columns:
    - median imputation
    - standard scaling

    Categorical columns:
    - most frequent imputation
    - one-hot encoding
    """

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def prepare_train_test_data(df, test_size=0.2):
    """
    Clean data, split into train/test sets, and build preprocessing object.
    """

    df_clean = clean_data(df)
    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    cleaned = clean_data(df)

    print(cleaned.head())
    print(cleaned[TARGET_COLUMN].value_counts(normalize=True))