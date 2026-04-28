import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X_test)

    # Some models support predict_proba, which is needed for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc,
    }

    print(f"\n===== {model_name} Results =====")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall:   {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return results


def save_confusion_matrix(model, X_test, y_test, output_path, model_name="Model"):
    """
    Save confusion matrix as an image.
    """

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()