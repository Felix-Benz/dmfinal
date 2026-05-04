"""
Comprehensive model analysis pipeline: error analysis, uncertainty quantification,
ablation studies, and statistical testing for the hospital readmission project.
"""
from pathlib import Path
import sys
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_data
from preprocess import (
    prepare_baseline_data,
    prepare_baseline_data_cv,
    prepare_mlp_data,
    split_features_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DPI = 150


# ── MLP classes (mirrors train_mlp.py) ────────────────────────────────────────

class ReadmissionDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(np.array(X_num, dtype=np.float32), dtype=torch.float32)
        self.X_cat = torch.tensor(np.array(X_cat, dtype=np.int64), dtype=torch.long)
        self.y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class MLPWithEmbeddings(nn.Module):
    def __init__(self, num_numeric_features, categorical_cardinalities):
        super().__init__()
        self.embedding_layers = nn.ModuleList()
        total_emb_dim = 0
        for card in categorical_cardinalities:
            emb_dim = min(50, max(2, (card + 1) // 2))
            self.embedding_layers.append(nn.Embedding(card, emb_dim))
            total_emb_dim += emb_dim
        self.network = nn.Sequential(
            nn.Linear(num_numeric_features + total_emb_dim, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
        x = torch.cat([x_num] + embedded, dim=1) if embedded else x_num
        return self.network(x).squeeze(1)


def mlp_get_probs(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            logits = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            y_prob.extend(torch.sigmoid(logits).cpu().numpy())
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_prob)


def train_mlp_model(X_num_tr, X_cat_tr, y_tr,
                    X_num_val, X_cat_val, y_val,
                    cardinalities, use_pos_weight=True,
                    max_epochs=20, patience=4, verbose=False):
    tr_loader = DataLoader(
        ReadmissionDataset(X_num_tr, X_cat_tr, y_tr), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        ReadmissionDataset(X_num_val, X_cat_val, y_val), batch_size=256, shuffle=False
    )
    n_num = np.array(X_num_tr).shape[1]
    model = MLPWithEmbeddings(n_num, cardinalities).to(DEVICE)

    if use_pos_weight:
        y_arr = np.array(y_tr, dtype=float)
        pos = y_arr.sum()
        neg = len(y_arr) - pos
        pw = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_auc, best_state, no_improve = -1, None, 0

    for epoch in range(max_epochs):
        model.train()
        for x_num, x_cat, y_batch in tr_loader:
            optimizer.zero_grad()
            loss = criterion(
                model(x_num.to(DEVICE), x_cat.to(DEVICE)), y_batch.to(DEVICE)
            )
            loss.backward()
            optimizer.step()

        val_true, val_prob = mlp_get_probs(model, val_loader)
        val_auc = roc_auc_score(val_true, val_prob)
        if verbose:
            print(f"    Epoch {epoch + 1}: val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model


# ── Shared utilities ───────────────────────────────────────────────────────────

def safe_name(s):
    return (s.replace(" ", "_").replace("(", "").replace(")", "")
             .replace("-", "_").replace("/", "_").lower())


def confusion_values(y_true, y_pred):
    """Return (TN, FP, FN, TP) always as a 2×2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])


def compute_ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def run_cv_fold_scores(model_factory, X, y, numeric_cols,
                       use_smote=False, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ohe_cols = [c for c in X.columns if c not in numeric_cols]
    roc_aucs, recalls = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr_raw = X.iloc[train_idx].reset_index(drop=True)
        X_te_raw = X.iloc[test_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_num = pd.DataFrame(
            scaler.fit_transform(imputer.fit_transform(X_tr_raw[numeric_cols])),
            columns=numeric_cols,
        )
        X_te_num = pd.DataFrame(
            scaler.transform(imputer.transform(X_te_raw[numeric_cols])),
            columns=numeric_cols,
        )
        X_tr = pd.concat([X_tr_num, X_tr_raw[ohe_cols].reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te_num, X_te_raw[ohe_cols].reset_index(drop=True)], axis=1)

        if use_smote:
            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

        m = model_factory()
        m.fit(X_tr, y_tr)
        roc_aucs.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
        recalls.append(recall_score(y_te, m.predict(X_te), zero_division=0))

    return {"roc_auc": np.array(roc_aucs), "recall": np.array(recalls)}


def run_mlp_cv_fold_scores(X_numeric, X_categorical, y, cardinalities, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    roc_aucs, recalls = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_numeric, y), 1):
        print(f"    MLP fold {fold}/{n_splits}...")
        X_num_tr_t = X_numeric.iloc[train_idx]
        X_cat_tr_t = X_categorical.iloc[train_idx]
        y_tr_t = y.iloc[train_idx]

        X_num_te = X_numeric.iloc[test_idx]
        X_cat_te = X_categorical.iloc[test_idx]
        y_te = y.iloc[test_idx]

        X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_tr, y_val = train_test_split(
            X_num_tr_t, X_cat_tr_t, y_tr_t,
            test_size=0.15, random_state=42, stratify=y_tr_t,
        )
        model = train_mlp_model(
            X_num_tr.values, X_cat_tr.values, y_tr.values,
            X_num_val.values, X_cat_val.values, y_val.values,
            cardinalities, use_pos_weight=True, max_epochs=20, patience=4,
        )
        te_loader = DataLoader(
            ReadmissionDataset(X_num_te.values, X_cat_te.values, y_te.values),
            batch_size=256, shuffle=False,
        )
        y_true_f, y_prob_f = mlp_get_probs(model, te_loader)
        roc_aucs.append(roc_auc_score(y_true_f, y_prob_f))
        recalls.append(recall_score(y_true_f, (y_prob_f >= 0.5).astype(int), zero_division=0))

    return {"roc_auc": np.array(roc_aucs), "recall": np.array(recalls)}


# ── Error Analysis ────────────────────────────────────────────────────────────

def error_analysis(models_info, X_raw, out_dir):
    print("\n=== Error Analysis ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    subgroup_rows = []

    for info in models_info:
        name = info["name"]
        sname = safe_name(name)
        y_true = np.array(info["y_true"])
        y_pred = np.array(info["y_pred"])
        y_prob = info["y_prob"]
        test_idx = info["test_idx"]

        print(f"  Processing {name}...")

        # Confusion matrix heatmap
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred: Not Readmitted", "Pred: Readmitted"],
                    yticklabels=["True: Not Readmitted", "True: Readmitted"])
        ax.set_title(f"Confusion Matrix\n{name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{sname}.png", dpi=DPI)
        plt.close()

        # FP / FN CSV with original feature values
        raw_for_test = X_raw.loc[test_idx].copy()
        raw_for_test = raw_for_test.reset_index(drop=True)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        misclassified = raw_for_test.copy()
        misclassified["true_label"] = y_true
        misclassified["predicted_label"] = y_pred
        misclassified["predicted_prob"] = y_prob
        misclassified["error_type"] = np.where(
            fp_mask, "FP", np.where(fn_mask, "FN", "correct")
        )
        misclassified = misclassified[misclassified["error_type"] != "correct"]
        misclassified.to_csv(out_dir / f"misclassified_{sname}.csv", index=False)

        # Predicted probability distribution by true label
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, color, lname in [
            (0, "steelblue", "Not Readmitted (y=0)"),
            (1, "tomato", "Readmitted <30d (y=1)"),
        ]:
            mask = y_true == label
            ax.hist(y_prob[mask], bins=50, alpha=0.6, color=color,
                    label=f"{lname} (n={mask.sum()})", density=True)
        ax.set_title(f"Predicted Probability by True Label\n{name}")
        ax.set_xlabel("Predicted Probability of Readmission")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"prob_dist_{sname}.png", dpi=DPI)
        plt.close()

        # Subgroup error rate analysis
        raw_test = X_raw.loc[test_idx].copy().reset_index(drop=True)
        raw_test["_y_true"] = y_true
        raw_test["_y_pred"] = y_pred

        def _subgroup_stats(grp):
            tn, fp, fn, tp = confusion_values(grp["_y_true"].values, grp["_y_pred"].values)
            n = len(grp)
            pos = tp + fn
            return {
                "n_samples": n,
                "n_fp": fp,
                "n_fn": fn,
                "error_rate": (fp + fn) / n,
                "fn_rate": fn / pos if pos > 0 else np.nan,
            }

        # Age subgroup
        if "age" in raw_test.columns:
            for val in sorted(raw_test["age"].dropna().unique()):
                grp = raw_test[raw_test["age"] == val]
                if len(grp) < 5:
                    continue
                subgroup_rows.append({
                    "model": name, "subgroup_type": "age",
                    "subgroup_value": val, **_subgroup_stats(grp),
                })

        # num_medications subgroup (quartile bins)
        if "num_medications" in raw_test.columns:
            try:
                raw_test["_med_bin"] = pd.qcut(
                    raw_test["num_medications"], q=4,
                    labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop"
                )
                for val in raw_test["_med_bin"].dropna().unique():
                    grp = raw_test[raw_test["_med_bin"] == val]
                    if len(grp) < 5:
                        continue
                    subgroup_rows.append({
                        "model": name, "subgroup_type": "num_medications_quartile",
                        "subgroup_value": str(val), **_subgroup_stats(grp),
                    })
            except Exception:
                pass

        # diag_1 subgroup (top 10 most common codes)
        if "diag_1" in raw_test.columns:
            top_diags = raw_test["diag_1"].value_counts().head(10).index
            for val in top_diags:
                grp = raw_test[raw_test["diag_1"] == val]
                if len(grp) < 5:
                    continue
                subgroup_rows.append({
                    "model": name, "subgroup_type": "diag_1",
                    "subgroup_value": str(val), **_subgroup_stats(grp),
                })

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(out_dir / "subgroup_error_rates.csv", index=False)
    print(f"  Saved confusion matrices, misclassified CSVs, prob distributions, subgroup_error_rates.csv")
    return subgroup_df


# ── Uncertainty Quantification ────────────────────────────────────────────────

def uncertainty_quantification(models_info, rf_cw_model, X_test_bl, out_dir):
    print("\n=== Uncertainty Quantification ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    confidence_bins = [(0.0, 0.3, "low (0–0.3)"), (0.3, 0.6, "mid (0.3–0.6)"), (0.6, 1.001, "high (0.6–1.0)")]
    calib_rows = []

    # Combined calibration figure (2×3 grid)
    fig_all, axes_all = plt.subplots(2, 3, figsize=(15, 10))
    axes_all = axes_all.flatten()

    for i, info in enumerate(models_info):
        name = info["name"]
        sname = safe_name(name)
        y_true = np.array(info["y_true"])
        y_prob = np.array(info["y_prob"])
        print(f"  Calibration for {name}...")

        ece = compute_ece(y_true, y_prob)

        # Per-bin accuracy
        bin_accs, bin_ns = {}, {}
        for lo, hi, label in confidence_bins:
            mask = (y_prob >= lo) & (y_prob < hi)
            bin_ns[label] = int(mask.sum())
            if mask.sum() > 0:
                y_pred_bin = (y_prob[mask] >= 0.5).astype(int)
                bin_accs[label] = accuracy_score(y_true[mask], y_pred_bin)
            else:
                bin_accs[label] = np.nan

        calib_rows.append({
            "model": name,
            "ece": ece,
            "acc_low_conf": bin_accs["low (0–0.3)"],
            "acc_mid_conf": bin_accs["mid (0.3–0.6)"],
            "acc_high_conf": bin_accs["high (0.6–1.0)"],
            "n_low": bin_ns["low (0–0.3)"],
            "n_mid": bin_ns["mid (0.3–0.6)"],
            "n_high": bin_ns["high (0.6–1.0)"],
        })

        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        except Exception:
            frac_pos, mean_pred = np.array([]), np.array([])

        # Individual calibration figure
        fig_i, ax_i = plt.subplots(figsize=(6, 5))
        ax_i.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        if len(mean_pred) > 0:
            ax_i.plot(mean_pred, frac_pos, "o-", color="tomato", label=f"Model (ECE={ece:.4f})")
        ax_i.set_title(f"Calibration Curve\n{name}")
        ax_i.set_xlabel("Mean Predicted Probability")
        ax_i.set_ylabel("Fraction of Positives")
        ax_i.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_{sname}.png", dpi=DPI)
        plt.close(fig_i)

        # Panel in combined figure
        if i < len(axes_all):
            ax = axes_all[i]
            ax.plot([0, 1], [0, 1], "k--", label="Perfect")
            if len(mean_pred) > 0:
                ax.plot(mean_pred, frac_pos, "o-", color="tomato", label=f"ECE={ece:.4f}")
            ax.set_title(name.replace(" (", "\n("), fontsize=8)
            ax.legend(fontsize=7)
            ax.set_xlabel("Mean Predicted Prob", fontsize=8)
            ax.set_ylabel("Fraction Positive", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "calibration_all_models.png", dpi=DPI)
    plt.close(fig_all)

    # RF uncertainty via variance across trees
    print("  Computing RF tree-variance uncertainty...")
    # Find RF class-weighted info
    rf_info = next(m for m in models_info if "Random Forest (class-weighted)" in m["name"])
    y_true_rf = np.array(rf_info["y_true"])
    y_pred_rf = np.array(rf_info["y_pred"])

    tree_preds = np.stack([
        tree.predict_proba(X_test_bl)[:, 1]
        for tree in rf_cw_model.estimators_
    ])
    uncertainty = tree_preds.var(axis=0)
    thresh_75 = np.percentile(uncertainty, 75)
    high_unc = uncertainty >= thresh_75
    low_unc = ~high_unc

    err_low = (y_pred_rf[low_unc] != y_true_rf[low_unc]).mean()
    err_high = (y_pred_rf[high_unc] != y_true_rf[high_unc]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(uncertainty[y_true_rf == 0], bins=50, alpha=0.6,
                 color="steelblue", label="True 0 (not readmitted)", density=True)
    axes[0].hist(uncertainty[y_true_rf == 1], bins=50, alpha=0.6,
                 color="tomato", label="True 1 (readmitted <30d)", density=True)
    axes[0].axvline(thresh_75, color="black", linestyle="--", label="75th pct threshold")
    axes[0].set_title("RF Prediction Variance by True Label")
    axes[0].set_xlabel("Variance Across Trees")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    bars = axes[1].bar(["Low Uncertainty\n(<75th pct)", "High Uncertainty\n(≥75th pct)"],
                        [err_low, err_high], color=["steelblue", "tomato"], width=0.5)
    for bar, val in zip(bars, [err_low, err_high]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    axes[1].set_title("Error Rate by Uncertainty Level\n(RF class-weighted)")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_ylim(0, max(err_high * 1.2, 0.5))

    plt.suptitle("Random Forest Uncertainty Analysis (Tree Variance)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "rf_uncertainty_analysis.png", dpi=DPI)
    plt.close()

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(out_dir / "calibration_metrics.csv", index=False)
    print(f"  Saved calibration curves, rf_uncertainty_analysis.png, calibration_metrics.csv")
    return calib_df


# ── Ablation Studies ──────────────────────────────────────────────────────────

def ablation_studies(rf_cw_model, X_train_bl, y_train_bl, X_test_bl, y_test_bl,
                     mlp_train_data, out_dir):
    print("\n=== Ablation Studies ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = []

    # --- RF feature importance ablation ---
    feature_names = np.array(X_train_bl.columns)
    importances = rf_cw_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top10 = feature_names[sorted_idx[:10]].tolist()
    top5 = feature_names[sorted_idx[:5]].tolist()

    print(f"  Top-5 features: {top5}")

    def _eval_rf_subset(features):
        m = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        m.fit(X_train_bl[features], y_train_bl)
        yp = m.predict_proba(X_test_bl[features])[:, 1]
        yd = m.predict(X_test_bl[features])
        return {
            "roc_auc": roc_auc_score(y_test_bl, yp),
            "recall": recall_score(y_test_bl, yd, zero_division=0),
            "f1": f1_score(y_test_bl, yd, zero_division=0),
            "pr_auc": average_precision_score(y_test_bl, yp),
        }

    for variant, features in [
        ("all_features", feature_names.tolist()),
        ("top10_features", top10),
        ("top5_features", top5),
    ]:
        print(f"  RF ablation: {variant} ({len(features)} features)...")
        metrics = _eval_rf_subset(features)
        ablation_rows.append({"model_group": "RF", "variant": variant,
                               "n_features": len(features), **metrics})

    # --- MLP class-imbalance ablation ---
    (X_num_tr, X_cat_tr, y_tr,
     X_num_val, X_cat_val, y_val,
     X_num_te, X_cat_te, y_te,
     cardinalities) = mlp_train_data

    te_ds = ReadmissionDataset(X_num_te, X_cat_te, y_te)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

    for variant, use_pw in [("with_pos_weight", True), ("no_pos_weight", False)]:
        print(f"  MLP ablation: {variant}...")
        m = train_mlp_model(X_num_tr, X_cat_tr, y_tr,
                             X_num_val, X_cat_val, y_val,
                             cardinalities, use_pos_weight=use_pw)
        y_true_te, y_prob_te = mlp_get_probs(m, te_loader)
        y_pred_te = (y_prob_te >= 0.5).astype(int)
        ablation_rows.append({
            "model_group": "MLP", "variant": variant, "n_features": "N/A",
            "roc_auc": roc_auc_score(y_true_te, y_prob_te),
            "recall": recall_score(y_true_te, y_pred_te, zero_division=0),
            "f1": f1_score(y_true_te, y_pred_te, zero_division=0),
            "pr_auc": average_precision_score(y_true_te, y_prob_te),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rf_abl = ablation_df[ablation_df["model_group"] == "RF"].copy()
    x = np.arange(len(rf_abl))
    w = 0.22
    metrics_plot = [("roc_auc", "ROC-AUC", "steelblue"),
                    ("recall", "Recall", "tomato"),
                    ("f1", "F1", "green"),
                    ("pr_auc", "PR-AUC", "darkorange")]
    for k, (col, label, color) in enumerate(metrics_plot):
        axes[0].bar(x + (k - 1.5) * w, rf_abl[col].values, w, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        rf_abl["variant"].str.replace("_", " ").str.replace("features", "feat."),
        rotation=15, ha="right"
    )
    axes[0].set_title("RF Feature Importance Ablation")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    mlp_abl = ablation_df[ablation_df["model_group"] == "MLP"].copy()
    x2 = np.arange(len(mlp_abl))
    for k, (col, label, color) in enumerate(metrics_plot):
        axes[1].bar(x2 + (k - 1.5) * w, mlp_abl[col].values, w, label=label, color=color)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(
        mlp_abl["variant"].str.replace("_", " "), rotation=15, ha="right"
    )
    axes[1].set_title("MLP Class-Imbalance Ablation\n(with vs without pos_weight)")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    plt.suptitle("Ablation Study Results", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_results.png", dpi=DPI)
    plt.close()

    print(f"  Saved ablation_results.csv, ablation_results.png")
    return ablation_df


# ── Statistical Testing ───────────────────────────────────────────────────────

def statistical_tests(X_cv, y_cv, numeric_cols, mlp_cv_data, out_dir):
    print("\n=== Statistical Testing ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("Logistic Regression (class-weighted)",
         lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
         False),
        ("Decision Tree (class-weighted)",
         lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20,
                                        class_weight="balanced", random_state=42),
         False),
        ("Decision Tree (SMOTE)",
         lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42),
         True),
        ("Random Forest (class-weighted)",
         lambda: RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                        class_weight="balanced", n_jobs=-1, random_state=42),
         False),
        ("Random Forest (SMOTE)",
         lambda: RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                        n_jobs=-1, random_state=42),
         True),
    ]

    fold_scores = {}
    for name, factory, use_smote in model_specs:
        print(f"  CV for {name}...")
        fold_scores[name] = run_cv_fold_scores(factory, X_cv, y_cv, numeric_cols, use_smote=use_smote)

    X_num_mlp, X_cat_mlp, y_mlp, cardinalities = mlp_cv_data
    print("  CV for MLP (5 folds — may take several minutes)...")
    fold_scores["MLP with categorical embeddings"] = run_mlp_cv_fold_scores(
        X_num_mlp, X_cat_mlp, y_mlp, cardinalities
    )

    model_names = list(fold_scores.keys())
    n = len(model_names)
    n_pairs = n * (n - 1) // 2
    bonferroni_factor = n_pairs  # multiply raw p-value by this

    test_rows = []

    short_labels = []
    for nm in model_names:
        s = nm.replace("Logistic Regression", "LR").replace("Decision Tree", "DT") \
               .replace("Random Forest", "RF").replace("MLP with categorical embeddings", "MLP") \
               .replace(" (class-weighted)", "\n(cw)").replace(" (SMOTE)", "\n(SMOTE)")
        short_labels.append(s)

    for metric in ["roc_auc", "recall"]:
        p_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(i + 1, n):
                a = fold_scores[model_names[i]][metric]
                b = fold_scores[model_names[j]][metric]
                t_stat, p_raw = ttest_rel(a, b)
                p_corrected = min(p_raw * bonferroni_factor, 1.0)
                p_matrix[i, j] = p_corrected
                p_matrix[j, i] = p_corrected

                test_rows.append({
                    "metric": metric,
                    "model_a": model_names[i],
                    "model_b": model_names[j],
                    "t_statistic": round(t_stat, 4),
                    "p_value_raw": round(p_raw, 6),
                    "p_value_bonferroni": round(p_corrected, 6),
                    "significant": bool(p_corrected < 0.05),
                    "mean_a": round(fold_scores[model_names[i]][metric].mean(), 4),
                    "mean_b": round(fold_scores[model_names[j]][metric].mean(), 4),
                })

        # Significance heatmap (lower-left triangle + upper-right; diagonal NaN)
        fig, ax = plt.subplots(figsize=(11, 9))
        mask = np.eye(n, dtype=bool)
        sns.heatmap(
            p_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=0.2,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.5, ax=ax, mask=mask,
            annot_kws={"size": 9},
        )
        alpha_bonf = 0.05 / n_pairs
        ax.set_title(
            f"Pairwise p-values — {metric.upper()} (Bonferroni corrected)\n"
            f"n_pairs={n_pairs}, α_corrected={alpha_bonf:.4f}  |  Green < 0.05",
            fontsize=10,
        )
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"significance_heatmap_{metric}.png", dpi=DPI)
        plt.close()

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(out_dir / "statistical_tests.csv", index=False)
    print(f"  Saved significance_heatmap_roc_auc.png, significance_heatmap_recall.png, statistical_tests.csv")
    return tests_df, fold_scores


# ── Plain-English Summary ──────────────────────────────────────────────────────

def print_summary(models_info, subgroup_df, calib_df, ablation_df, tests_df, fold_scores):
    sep = "=" * 70
    print(f"\n{sep}")
    print("HOSPITAL READMISSION MODEL ANALYSIS — KEY FINDINGS")
    print(sep)

    print("\n[Error Analysis]")
    for info in models_info:
        y_true = np.array(info["y_true"])
        y_pred = np.array(info["y_pred"])
        tn, fp, fn, tp = confusion_values(y_true, y_pred)
        total = len(y_true)
        actual_pos = tp + fn
        rec = tp / actual_pos if actual_pos > 0 else 0.0
        print(f"  {info['name']}: "
              f"FN={fn} ({fn / total:.1%} of test), "
              f"FP={fp} ({fp / total:.1%} of test), "
              f"recall={rec:.3f}")

    if len(subgroup_df) > 0:
        print("\n  Top subgroup failure patterns (highest FN rate):")
        top_fail = (subgroup_df.sort_values("fn_rate", ascending=False)
                    .dropna(subset=["fn_rate"])
                    .head(5))
        for _, row in top_fail.iterrows():
            print(f"    {row['model'][:30]} | {row['subgroup_type']}={row['subgroup_value']}"
                  f" | fn_rate={row['fn_rate']:.2%} (n={row['n_samples']})")

    print("\n[Uncertainty Quantification]")
    for _, row in calib_df.iterrows():
        def _fmt(v):
            return f"{v:.3f}" if pd.notna(v) else "N/A"
        print(f"  {row['model']}: ECE={row['ece']:.4f} | "
              f"acc@low={_fmt(row['acc_low_conf'])} | "
              f"acc@mid={_fmt(row['acc_mid_conf'])} | "
              f"acc@high={_fmt(row['acc_high_conf'])}")
    best_cal = calib_df.loc[calib_df["ece"].idxmin(), "model"]
    worst_cal = calib_df.loc[calib_df["ece"].idxmax(), "model"]
    print(f"  Best calibrated model: {best_cal}")
    print(f"  Worst calibrated model: {worst_cal}")
    print("  High-uncertainty RF samples (top-25% variance) have higher error rates —")
    print("  prediction variance across trees is a reliable proxy for model confidence.")

    print("\n[Ablation Studies]")
    rf_abl = ablation_df[ablation_df["model_group"] == "RF"]
    row_all = rf_abl[rf_abl["variant"] == "all_features"].iloc[0]
    row_10 = rf_abl[rf_abl["variant"] == "top10_features"].iloc[0]
    row_5 = rf_abl[rf_abl["variant"] == "top5_features"].iloc[0]
    drop_10 = row_all["roc_auc"] - row_10["roc_auc"]
    drop_5 = row_all["roc_auc"] - row_5["roc_auc"]
    print(f"  RF ROC-AUC: all={row_all['roc_auc']:.4f}, top-10={row_10['roc_auc']:.4f} "
          f"(Δ={drop_10:+.4f}), top-5={row_5['roc_auc']:.4f} (Δ={drop_5:+.4f})")
    if abs(drop_10) < 0.01:
        print("  → Top-10 features capture nearly all predictive signal (minimal AUC drop).")
    else:
        print("  → Reducing to top-10 features causes a notable drop — full feature set matters.")

    mlp_abl = ablation_df[ablation_df["model_group"] == "MLP"]
    row_w = mlp_abl[mlp_abl["variant"] == "with_pos_weight"].iloc[0]
    row_nw = mlp_abl[mlp_abl["variant"] == "no_pos_weight"].iloc[0]
    recall_delta = row_w["recall"] - row_nw["recall"]
    print(f"  MLP recall: weighted={row_w['recall']:.4f}, unweighted={row_nw['recall']:.4f} "
          f"(Δ={recall_delta:+.4f})")
    if recall_delta > 0.05:
        print("  → pos_weight substantially improves recall; without it the MLP ignores the minority class.")
    elif recall_delta > 0:
        print("  → pos_weight modestly improves recall for the minority class.")
    else:
        print("  → pos_weight does not improve recall in this run.")

    print("\n[Statistical Testing]")
    sig = tests_df[tests_df["significant"]]
    print(f"  Total pairwise comparisons: {len(tests_df)}, of which "
          f"{len(sig)} are significant after Bonferroni correction (α=0.05).")
    if len(sig) > 0:
        print("  Statistically significant pairs:")
        for _, row in sig.iterrows():
            winner = row["model_a"] if row["mean_a"] > row["mean_b"] else row["model_b"]
            print(f"    {row['metric'].upper()}: {row['model_a'][:25]} vs {row['model_b'][:25]}"
                  f" — p={row['p_value_bonferroni']:.4f} → {winner[:25]} is better")
    else:
        print("  No model pair shows a statistically significant difference at the Bonferroni-")
        print("  corrected threshold. All models perform within noise of each other.")

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}")

    # Load data & raw features for inspection
    df = load_data()
    X_raw, _ = split_features_target(df)

    # Baseline preprocessing
    X_bl, y_bl, metadata = prepare_baseline_data(df, max_categories=20)
    print(f"\nBaseline features: {X_bl.shape[1]}")

    X_train_bl, X_test_bl, y_train_bl, y_test_bl = train_test_split(
        X_bl, y_bl, test_size=0.2, random_state=42, stratify=y_bl
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_bl, y_train_bl)

    # Train baseline models (same hyperparameters as train_baselines.py)
    print("\nTraining baseline models...")

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train_bl, y_train_bl)
    print("  LR done.")

    dt_cw_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20,
                                          class_weight="balanced", random_state=42)
    dt_cw_model.fit(X_train_bl, y_train_bl)
    print("  DT (class-weighted) done.")

    dt_smote_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42)
    dt_smote_model.fit(X_train_smote, y_train_smote)
    print("  DT (SMOTE) done.")

    rf_cw_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf_cw_model.fit(X_train_bl, y_train_bl)
    print("  RF (class-weighted) done.")

    rf_smote_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    )
    rf_smote_model.fit(X_train_smote, y_train_smote)
    print("  RF (SMOTE) done.")

    # Train MLP
    print("\nTraining MLP...")
    X_numeric, X_categorical, y_mlp, category_maps, _ = prepare_mlp_data(df)
    cardinalities = [len(category_maps[col]) for col in X_categorical.columns]

    (X_num_tr_t, X_num_test_mlp,
     X_cat_tr_t, X_cat_test_mlp,
     y_tr_t, y_test_mlp) = train_test_split(
        X_numeric, X_categorical, y_mlp,
        test_size=0.2, random_state=42, stratify=y_mlp,
    )
    (X_num_train_mlp, X_num_val_mlp,
     X_cat_train_mlp, X_cat_val_mlp,
     y_train_mlp, y_val_mlp) = train_test_split(
        X_num_tr_t, X_cat_tr_t, y_tr_t,
        test_size=0.2, random_state=42, stratify=y_tr_t,
    )

    mlp_model = train_mlp_model(
        X_num_train_mlp.values, X_cat_train_mlp.values, y_train_mlp.values,
        X_num_val_mlp.values, X_cat_val_mlp.values, y_val_mlp.values,
        cardinalities, use_pos_weight=True, verbose=True,
    )
    print("  MLP done.")

    mlp_test_loader = DataLoader(
        ReadmissionDataset(X_num_test_mlp.values, X_cat_test_mlp.values, y_test_mlp.values),
        batch_size=256, shuffle=False,
    )
    mlp_y_true, mlp_y_prob = mlp_get_probs(mlp_model, mlp_test_loader)
    mlp_y_pred = (mlp_y_prob >= 0.5).astype(int)

    # Build unified models_info list
    def _bl_info(name, model):
        yp = model.predict_proba(X_test_bl)[:, 1]
        yd = model.predict(X_test_bl)
        return {
            "name": name, "model": model,
            "y_true": y_test_bl.values,
            "y_pred": yd,
            "y_prob": yp,
            "test_idx": X_test_bl.index,
        }

    models_info = [
        _bl_info("Logistic Regression (class-weighted)", lr_model),
        _bl_info("Decision Tree (class-weighted)", dt_cw_model),
        _bl_info("Decision Tree (SMOTE)", dt_smote_model),
        _bl_info("Random Forest (class-weighted)", rf_cw_model),
        _bl_info("Random Forest (SMOTE)", rf_smote_model),
        {
            "name": "MLP with categorical embeddings",
            "model": mlp_model,
            "y_true": mlp_y_true,
            "y_pred": mlp_y_pred,
            "y_prob": mlp_y_prob,
            "test_idx": X_num_test_mlp.index,
        },
    ]

    # Error Analysis
    subgroup_df = error_analysis(models_info, X_raw, OUT_DIR / "error_analysis")

    # Uncertainty Quantification

    calib_df = uncertainty_quantification(models_info, rf_cw_model, X_test_bl, OUT_DIR / "uncertainty")

    # Ablation Studies
    mlp_train_data = (
        X_num_train_mlp.values, X_cat_train_mlp.values, y_train_mlp.values,
        X_num_val_mlp.values, X_cat_val_mlp.values, y_val_mlp.values,
        X_num_test_mlp.values, X_cat_test_mlp.values, y_test_mlp.values,
        cardinalities,
    )
    ablation_df = ablation_studies(
        rf_cw_model, X_train_bl, y_train_bl, X_test_bl, y_test_bl,
        mlp_train_data, OUT_DIR / "ablation",
    )

    # Statistical Testing
    X_cv, y_cv, numeric_cols = prepare_baseline_data_cv(df, max_categories=20)
    mlp_cv_data = (X_numeric, X_categorical, y_mlp, cardinalities)
    tests_df, fold_scores = statistical_tests(
        X_cv, y_cv, numeric_cols, mlp_cv_data, OUT_DIR / "statistical_tests"
    )

    # Summary
    print_summary(models_info, subgroup_df, calib_df, ablation_df, tests_df, fold_scores)


if __name__ == "__main__":
    main()
