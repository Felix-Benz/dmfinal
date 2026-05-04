from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_data
from preprocess import prepare_baseline_data, prepare_baseline_data_cv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate a trained classification model and return a dictionary of metrics.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = y_pred

    metrics = {
        "model": name,
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "pr_auc": average_precision_score(y_test, y_scores),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics in a readable format.
    """
    print(f"\n=== {metrics['model']} ===")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def run_cv(name, model_factory, X, y, numeric_cols, use_smote=False, n_splits=5):
    """
    Run stratified k-fold cross-validation for a given model and return mean ± std metrics.

    Imputation and scaling are fit on each training fold to prevent leakage.
    SMOTE is applied inside each fold when use_smote=True.
    model_factory is a zero-argument callable that returns a fresh model instance.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ohe_cols = [c for c in X.columns if c not in numeric_cols]
    fold_metrics = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        # Fit imputation and scaling on this training fold only
        num_imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_num = pd.DataFrame(
            scaler.fit_transform(num_imputer.fit_transform(X_train_raw[numeric_cols])),
            columns=numeric_cols,
        )
        X_test_num = pd.DataFrame(
            scaler.transform(num_imputer.transform(X_test_raw[numeric_cols])),
            columns=numeric_cols,
        )

        X_train_fold = pd.concat([X_train_num, X_train_raw[ohe_cols]], axis=1)
        X_test_fold = pd.concat([X_test_num, X_test_raw[ohe_cols]], axis=1)

        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_fold, y_train = smote.fit_resample(X_train_fold, y_train)

        model = model_factory()
        model.fit(X_train_fold, y_train)

        y_pred = model.predict(X_test_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]

        fold_metrics.append({
            "recall":  recall_score(y_test, y_pred, zero_division=0),
            "f1":      f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_scores),
            "pr_auc":  average_precision_score(y_test, y_scores),
        })

    metrics_df = pd.DataFrame(fold_metrics)
    result = {"model": name}
    for col in metrics_df.columns:
        result[f"{col}_mean"] = metrics_df[col].mean()
        result[f"{col}_std"] = metrics_df[col].std()

    return result


def print_cv_metrics(metrics):
    """
    Print cross-validation mean ± std metrics.
    """
    print(f"\n=== {metrics['model']} (5-fold CV) ===")
    print(f"Recall:   {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"F1:       {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc_mean']:.4f} ± {metrics['pr_auc_std']:.4f}")


def main():
    # Load and preprocess data
    df = load_data()
    X, y, metadata = prepare_baseline_data(df, max_categories=20)

    print("Baseline preprocessing complete.")
    print(f"Final feature count: {metadata['num_features_final']}")
    print(f"Numeric columns: {len(metadata['numeric_cols'])}")
    print(f"Selected categorical columns: {len(metadata['selected_categorical_cols'])}")
    print(f"Skipped high-cardinality columns: {len(metadata['skipped_high_cardinality_cols'])}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split complete.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nSMOTE applied to training data.")
    print(f"Original training size: {len(X_train)}")
    print(f"Resampled training size: {len(X_train_smote)}")

    results = []

    # 1. Logistic Regression baseline
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    logistic_model.fit(X_train, y_train)
    logistic_metrics = evaluate_model("Logistic Regression (class-weighted)", logistic_model, X_test, y_test)
    print_metrics(logistic_metrics)
    results.append(logistic_metrics)

    # 2. Decision Tree with class weights
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model("Decision Tree (class-weighted)", dt_model, X_test, y_test)
    print_metrics(dt_metrics)
    results.append(dt_metrics)

    # 3. Decision Tree with SMOTE
    dt_smote_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
    )
    dt_smote_model.fit(X_train_smote, y_train_smote)
    dt_smote_metrics = evaluate_model("Decision Tree (SMOTE)", dt_smote_model, X_test, y_test)
    print_metrics(dt_smote_metrics)
    results.append(dt_smote_metrics)

    # 4. Random Forest with class weights
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest (class-weighted)", rf_model, X_test, y_test)
    print_metrics(rf_metrics)
    results.append(rf_metrics)

    # 5. Random Forest with SMOTE
    rf_smote_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf_smote_model.fit(X_train_smote, y_train_smote)
    rf_smote_metrics = evaluate_model("Random Forest (SMOTE)", rf_smote_model, X_test, y_test)
    print_metrics(rf_smote_metrics)
    results.append(rf_smote_metrics)

    # Save summary results
    results_table = pd.DataFrame(
        [
            {
                "model": r["model"],
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": r["f1"],
                "accuracy": r["accuracy"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
            }
            for r in results
        ]
    )

    results_path = OUTPUTS_DIR / "baseline_results.csv"
    results_table.to_csv(results_path, index=False)

    print("\nSaved baseline results to:")
    print(results_path)

    print("\nSummary table:")
    print(results_table.sort_values(by="recall", ascending=False).to_string(index=False))

    # Run 5-fold stratified cross-validation on all baseline models
    print("\n\nRunning 5-fold stratified cross-validation...")
    X_cv, y_cv, numeric_cols = prepare_baseline_data_cv(df, max_categories=20)

    cv_results = []

    # 1. Logistic Regression CV
    lr_cv = run_cv(
        "Logistic Regression (class-weighted)",
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_cv, y_cv, numeric_cols,
    )
    print_cv_metrics(lr_cv)
    cv_results.append(lr_cv)

    # 2. Decision Tree (class-weighted) CV
    dt_cv = run_cv(
        "Decision Tree (class-weighted)",
        lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=42),
        X_cv, y_cv, numeric_cols,
    )
    print_cv_metrics(dt_cv)
    cv_results.append(dt_cv)

    # 3. Decision Tree (SMOTE) CV
    dt_smote_cv = run_cv(
        "Decision Tree (SMOTE)",
        lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42),
        X_cv, y_cv, numeric_cols, use_smote=True,
    )
    print_cv_metrics(dt_smote_cv)
    cv_results.append(dt_smote_cv)

    # 4. Random Forest (class-weighted) CV
    rf_cv = run_cv(
        "Random Forest (class-weighted)",
        lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        X_cv, y_cv, numeric_cols,
    )
    print_cv_metrics(rf_cv)
    cv_results.append(rf_cv)

    # 5. Random Forest (SMOTE) CV
    rf_smote_cv = run_cv(
        "Random Forest (SMOTE)",
        lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=42,
        ),
        X_cv, y_cv, numeric_cols, use_smote=True,
    )
    print_cv_metrics(rf_smote_cv)
    cv_results.append(rf_smote_cv)

    cv_results_table = pd.DataFrame(cv_results)
    cv_results_path = OUTPUTS_DIR / "baseline_cv_results.csv"
    cv_results_table.to_csv(cv_results_path, index=False)

    print("\nSaved CV results to:")
    print(cv_results_path)

    print("\nCV summary table:")
    print(cv_results_table.sort_values(by="recall_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()