from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset

from data_loader import load_data
from preprocess import prepare_mlp_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReadmissionDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class MLPWithEmbeddings(nn.Module):
    def __init__(self, num_numeric_features, categorical_cardinalities):
        super().__init__()

        self.embedding_layers = nn.ModuleList()
        total_embedding_dim = 0

        for cardinality in categorical_cardinalities:
            emb_dim = min(50, max(2, (cardinality + 1) // 2))
            self.embedding_layers.append(nn.Embedding(cardinality, emb_dim))
            total_embedding_dim += emb_dim

        input_dim = num_numeric_features + total_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        embedded_features = []

        for i, emb_layer in enumerate(self.embedding_layers):
            embedded_features.append(emb_layer(x_cat[:, i]))

        if embedded_features:
            x_emb = torch.cat(embedded_features, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        return self.network(x).squeeze(1)


def collect_predictions(model, dataloader):
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_num, x_cat, y in dataloader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)

            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits)

            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_probs)


def evaluate_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_metrics(metrics, threshold):
    print(f"\n=== MLP with Categorical Embeddings (threshold={threshold:.2f}) ===")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.30, 0.71, 0.05)

    best_threshold = 0.5
    best_metrics = None
    best_score = -1

    for threshold in thresholds:
        metrics = evaluate_from_probs(y_true, y_prob, threshold=threshold)

        # prioritize F1, then recall
        score = metrics["f1"] + 0.2 * metrics["recall"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def run_mlp_cv(X_numeric, X_categorical, y, categorical_cardinalities, n_splits=5):
    """
    Run stratified k-fold cross-validation for the MLP with categorical embeddings.

    Uses the same architecture and hyperparameters as the single-split run.
    Each fold gets a fresh model trained from scratch, with an inner val split
    for early stopping. Reports mean ± std at the default 0.5 threshold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_numeric, y), 1):
        print(f"\nFold {fold}/{n_splits}")

        X_num_train_temp = X_numeric.iloc[train_idx]
        X_cat_train_temp = X_categorical.iloc[train_idx]
        y_train_temp = y.iloc[train_idx]

        X_num_test = X_numeric.iloc[test_idx]
        X_cat_test = X_categorical.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Inner train/val split for early stopping
        X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
            X_num_train_temp, X_cat_train_temp, y_train_temp,
            test_size=0.15, random_state=42, stratify=y_train_temp,
        )

        train_dataset = ReadmissionDataset(X_num_train.values, X_cat_train.values, y_train.values)
        val_dataset = ReadmissionDataset(X_num_val.values, X_cat_val.values, y_val.values)
        test_dataset = ReadmissionDataset(X_num_test.values, X_cat_test.values, y_test.values)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        model = MLPWithEmbeddings(
            num_numeric_features=X_numeric.shape[1],
            categorical_cardinalities=categorical_cardinalities,
        ).to(DEVICE)

        positive_count = y_train.sum()
        negative_count = len(y_train) - positive_count
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_auc = -1
        best_model_state = None
        patience = 4
        epochs_without_improvement = 0

        for epoch in range(20):
            model.train()
            total_loss = 0.0

            for x_num, x_cat, y_batch in train_loader:
                x_num = x_num.to(DEVICE)
                x_cat = x_cat.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                logits = model(x_num, x_cat)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_true, val_prob = collect_predictions(model, val_loader)
            val_auc = roc_auc_score(val_true, val_prob)

            print(
                f"  Epoch {epoch + 1}/20 - "
                f"Loss: {avg_loss:.4f} - "
                f"Val ROC-AUC: {val_auc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("  Early stopping triggered.")
                break

        model.load_state_dict(best_model_state)

        test_true, test_prob = collect_predictions(model, test_loader)
        fold_result = evaluate_from_probs(test_true, test_prob, threshold=0.5)

        fold_metrics.append({
            "recall":  fold_result["recall"],
            "f1":      fold_result["f1"],
            "roc_auc": fold_result["roc_auc"],
            "pr_auc":  fold_result["pr_auc"],
        })

        print(
            f"  Fold {fold} test: recall={fold_metrics[-1]['recall']:.4f}  "
            f"f1={fold_metrics[-1]['f1']:.4f}  "
            f"roc_auc={fold_metrics[-1]['roc_auc']:.4f}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    result = {"model": "MLP with categorical embeddings"}
    for col in metrics_df.columns:
        result[f"{col}_mean"] = metrics_df[col].mean()
        result[f"{col}_std"] = metrics_df[col].std()

    return result


def main():
    print(f"Using device: {DEVICE}")

    df = load_data()
    X_numeric, X_categorical, y, category_maps, metadata = prepare_mlp_data(df)

    print("MLP preprocessing complete.")
    print(f"Numeric feature count: {len(metadata['numeric_cols'])}")
    print(f"Categorical feature count: {len(metadata['categorical_cols'])}")

    # First split: train_temp / test
    X_num_train_temp, X_num_test, X_cat_train_temp, X_cat_test, y_train_temp, y_test = train_test_split(
        X_numeric,
        X_categorical,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Second split: train / validation
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_num_train_temp,
        X_cat_train_temp,
        y_train_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_train_temp,
    )

    train_dataset = ReadmissionDataset(X_num_train.values, X_cat_train.values, y_train.values)
    val_dataset = ReadmissionDataset(X_num_val.values, X_cat_val.values, y_val.values)
    test_dataset = ReadmissionDataset(X_num_test.values, X_cat_test.values, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    categorical_cardinalities = [len(category_maps[col]) for col in X_categorical.columns]

    model = MLPWithEmbeddings(
        num_numeric_features=X_numeric.shape[1],
        categorical_cardinalities=categorical_cardinalities,
    ).to(DEVICE)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    patience = 4
    best_val_auc = -1
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_num, x_cat, y_batch in train_loader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_true, val_prob = collect_predictions(model, val_loader)
        val_metrics = evaluate_from_probs(val_true, val_prob, threshold=0.5)
        val_auc = val_metrics["roc_auc"]

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_state)

    # Use validation set to choose threshold
    val_true, val_prob = collect_predictions(model, val_loader)
    best_threshold, best_val_metrics = find_best_threshold(val_true, val_prob)

    print("\nBest validation threshold selected:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_val_metrics['f1']:.4f}")
    print(f"Validation Recall: {best_val_metrics['recall']:.4f}")

    # Final test evaluation at default threshold
    test_true, test_prob = collect_predictions(model, test_loader)

    default_metrics = evaluate_from_probs(test_true, test_prob, threshold=0.50)
    print("\nDefault threshold results:")
    print_metrics(default_metrics, 0.50)

    tuned_metrics = evaluate_from_probs(test_true, test_prob, threshold=best_threshold)
    print("\nTuned threshold results:")
    print_metrics(tuned_metrics, best_threshold)

    results_df = pd.DataFrame([
        {
            "model": "MLP with categorical embeddings",
            "threshold_type": "default",
            "threshold": 0.50,
            "recall": default_metrics["recall"],
            "precision": default_metrics["precision"],
            "f1": default_metrics["f1"],
            "accuracy": default_metrics["accuracy"],
            "roc_auc": default_metrics["roc_auc"],
            "pr_auc": default_metrics["pr_auc"],
        },
        {
            "model": "MLP with categorical embeddings",
            "threshold_type": "tuned",
            "threshold": best_threshold,
            "recall": tuned_metrics["recall"],
            "precision": tuned_metrics["precision"],
            "f1": tuned_metrics["f1"],
            "accuracy": tuned_metrics["accuracy"],
            "roc_auc": tuned_metrics["roc_auc"],
            "pr_auc": tuned_metrics["pr_auc"],
        }
    ])

    results_path = OUTPUTS_DIR / "mlp_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nSaved MLP results to:")
    print(results_path)

    # Run 5-fold cross-validation
    print("\n\nRunning 5-fold cross-validation for MLP...")
    cv_result = run_mlp_cv(X_numeric, X_categorical, y, categorical_cardinalities)

    print(f"\n=== MLP with Categorical Embeddings (5-fold CV) ===")
    print(f"Recall:   {cv_result['recall_mean']:.4f} ± {cv_result['recall_std']:.4f}")
    print(f"F1:       {cv_result['f1_mean']:.4f} ± {cv_result['f1_std']:.4f}")
    print(f"ROC-AUC:  {cv_result['roc_auc_mean']:.4f} ± {cv_result['roc_auc_std']:.4f}")
    print(f"PR-AUC:   {cv_result['pr_auc_mean']:.4f} ± {cv_result['pr_auc_std']:.4f}")

    mlp_cv_df = pd.DataFrame([cv_result])
    mlp_cv_path = OUTPUTS_DIR / "mlp_cv_results.csv"
    mlp_cv_df.to_csv(mlp_cv_path, index=False)

    print("\nSaved MLP CV results to:")
    print(mlp_cv_path)


if __name__ == "__main__":
    main()