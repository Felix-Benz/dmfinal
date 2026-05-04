from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Columns that should not be used as predictive features
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Target column
TARGET_COLUMN = "readmitted"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for EDA and shared preprocessing.
    
    - Replaces '?' with NaN
    - Drops identifier columns
    - Converts readmitted into binary target:
        <30 -> 1
        otherwise -> 0
    """
    df = df.copy()

    # Replace placeholder missing values
    df = df.replace("?", np.nan)

    # Drop obvious identifier columns if they exist
    existing_id_cols = [col for col in ID_COLUMNS if col in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)

    # Convert target to binary
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "<30").astype(int)

    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y after basic preprocessing.
    """
    df = preprocess_data(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def get_feature_types(X: pd.DataFrame):
    """
    Identify numeric and categorical columns.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def prepare_baseline_data(
    df: pd.DataFrame,
    max_categories: int = 20
):
    """
    Prepare data for baseline tree models.

    Strategy:
    - preprocess data
    - split X/y
    - identify numeric and categorical columns
    - impute missing values
    - standardize numeric features
    - one-hot encode only lower/moderate-cardinality categorical columns
    - leave high-cardinality categorical columns out for now

    Returns:
    - X_final: processed feature dataframe
    - y: target series
    - metadata: useful info for debugging/reporting
    """
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = get_feature_types(X)

    # Select only lower-cardinality categorical columns for one-hot encoding
    selected_cat_cols = []
    skipped_cat_cols = []

    for col in categorical_cols:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= max_categories:
            selected_cat_cols.append(col)
        else:
            skipped_cat_cols.append(col)

    # Numerical processing
    X_num = X[numeric_cols].copy()
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_num_imputed = pd.DataFrame(
            num_imputer.fit_transform(X_num),
            columns=numeric_cols,
            index=X.index,
        )
        X_num_scaled = pd.DataFrame(
            scaler.fit_transform(X_num_imputed),
            columns=numeric_cols,
            index=X.index,
        )
    else:
        X_num_scaled = pd.DataFrame(index=X.index)

    # Categorical processing for selected columns
    if len(selected_cat_cols) > 0:
        X_cat = X[selected_cat_cols].copy()
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imputed = pd.DataFrame(
            cat_imputer.fit_transform(X_cat),
            columns=selected_cat_cols,
            index=X.index,
        )
        X_cat_encoded = pd.get_dummies(
            X_cat_imputed,
            columns=selected_cat_cols,
            drop_first=False,
            dtype=int,
        )
    else:
        X_cat_encoded = pd.DataFrame(index=X.index)

    X_final = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "selected_categorical_cols": selected_cat_cols,
        "skipped_high_cardinality_cols": skipped_cat_cols,
        "num_features_final": X_final.shape[1],
    }

    return X_final, y, metadata


def prepare_mlp_data(df: pd.DataFrame, min_freq: int = 1):
    """
    Prepare data for a future MLP with categorical embeddings.

    Returns:
    - X_numeric: scaled numeric dataframe
    - X_categorical: dataframe of integer-encoded categorical columns
    - y: target series
    - category_maps: dict of category-to-index mappings
    - metadata: summary information

    Notes:
    - Missing categorical values are filled with 'MISSING'
    - Each categorical column is integer encoded starting at 0
    - This function does NOT build tensors yet; it prepares tabular inputs
    """
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = get_feature_types(X)

    # Numeric features
    X_num = X[numeric_cols].copy()
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_num_imputed = pd.DataFrame(
            num_imputer.fit_transform(X_num),
            columns=numeric_cols,
            index=X.index,
        )
        X_numeric = pd.DataFrame(
            scaler.fit_transform(X_num_imputed),
            columns=numeric_cols,
            index=X.index,
        )
    else:
        X_numeric = pd.DataFrame(index=X.index)

    # Categorical features for embeddings
    X_cat = X[categorical_cols].copy()
    X_categorical = pd.DataFrame(index=X.index)
    category_maps = {}

    for col in categorical_cols:
        filled = X_cat[col].fillna("MISSING").astype(str)

        # Optional rare-category filtering
        value_counts = filled.value_counts()
        rare_values = value_counts[value_counts < min_freq].index
        filled = filled.replace(rare_values, "RARE")

        categories = pd.Index(sorted(filled.unique()))
        mapping = {cat: idx for idx, cat in enumerate(categories)}

        X_categorical[col] = filled.map(mapping).astype(int)
        category_maps[col] = mapping

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "embedding_sizes": {
            col: len(mapping) for col, mapping in category_maps.items()
        },
    }

    return X_numeric, X_categorical, y, category_maps, metadata


def prepare_baseline_data_cv(df: pd.DataFrame, max_categories: int = 20):
    """
    Preprocessing for CV baselines.

    OHE is applied once on the full dataset so column structure is consistent
    across folds (no label info leaks from this).  Imputation and scaling are
    intentionally deferred — callers should fit them inside each CV fold via a
    Pipeline or ColumnTransformer.

    Returns:
    - X_final: OHE'd features with numeric columns still unimputed/unscaled
    - y: binary target series
    - numeric_cols: list of original numeric column names (for ColumnTransformer)
    """
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = get_feature_types(X)

    selected_cat_cols = [
        col for col in categorical_cols
        if X[col].nunique(dropna=True) <= max_categories
    ]

    # Fill categorical NaN with a constant so OHE produces consistent columns.
    X_cat = X[selected_cat_cols].fillna("MISSING").copy()
    X_cat_encoded = pd.get_dummies(
        X_cat, columns=selected_cat_cols, drop_first=False, dtype=int
    )

    X_num = X[numeric_cols].copy()
    X_final = pd.concat([X_num, X_cat_encoded], axis=1)

    return X_final, y, numeric_cols