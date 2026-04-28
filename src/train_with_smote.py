import pandas as pd
import joblib

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data
from preprocess import prepare_train_test_data
from evaluate import evaluate_model, save_confusion_matrix
from config import RANDOM_STATE, FIGURES_DIR, MODELS_DIR


def train_smote_model():
    """
    Train a Random Forest model using SMOTE on the training data.

    Important:
    SMOTE should only be applied to the training set, not the test set.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=100,
                    max_depth=12,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest with SMOTE...")
    model.fit(X_train, y_train)

    result = evaluate_model(
        model,
        X_test,
        y_test,
        model_name="Random Forest with SMOTE",
    )

    save_confusion_matrix(
        model,
        X_test,
        y_test,
        FIGURES_DIR / "random_forest_smote_confusion_matrix.png",
        model_name="Random Forest with SMOTE",
    )

    joblib.dump(model, MODELS_DIR / "random_forest_smote.joblib")

    pd.DataFrame([result]).to_csv(
        MODELS_DIR / "smote_results.csv",
        index=False,
    )


if __name__ == "__main__":
    train_smote_model()