import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data
from preprocess import prepare_train_test_data
from evaluate import evaluate_model, save_confusion_matrix
from config import RANDOM_STATE, FIGURES_DIR, MODELS_DIR


def train_models():
    """
    Train baseline classification models and compare performance.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=8,
        ),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, classifier in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        result = evaluate_model(
            pipeline,
            X_test,
            y_test,
            model_name=model_name,
        )

        results.append(result)

        safe_name = model_name.lower().replace(" ", "_")

        save_confusion_matrix(
            pipeline,
            X_test,
            y_test,
            FIGURES_DIR / f"{safe_name}_confusion_matrix.png",
            model_name=model_name,
        )

        joblib.dump(pipeline, MODELS_DIR / f"{safe_name}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODELS_DIR / "baseline_results.csv", index=False)

    print("\n===== Summary =====")
    print(results_df)


if __name__ == "__main__":
    train_models()