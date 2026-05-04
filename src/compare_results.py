from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main():
    baseline_path = OUTPUTS_DIR / "baseline_results.csv"
    mlp_path = OUTPUTS_DIR / "mlp_results.csv"

    baseline_df = pd.read_csv(baseline_path)
    mlp_df = pd.read_csv(mlp_path)

    comparison_df = pd.concat([baseline_df, mlp_df], ignore_index=True)

    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("Combined model comparison (single split):")
    print(comparison_df.to_string(index=False))

    print("\nSaved combined comparison to:")
    print(comparison_path)

    # Cross-validation summary (printed if both CV result files exist)
    baseline_cv_path = OUTPUTS_DIR / "baseline_cv_results.csv"
    mlp_cv_path = OUTPUTS_DIR / "mlp_cv_results.csv"

    if baseline_cv_path.exists() and mlp_cv_path.exists():
        baseline_cv_df = pd.read_csv(baseline_cv_path)
        mlp_cv_df = pd.read_csv(mlp_cv_path)

        cv_df = pd.concat([baseline_cv_df, mlp_cv_df], ignore_index=True)

        cv_comparison_path = OUTPUTS_DIR / "cv_comparison.csv"
        cv_df.to_csv(cv_comparison_path, index=False)

        print("\nCross-validation comparison (mean ± std):")
        print(cv_df.to_string(index=False))

        print("\nSaved CV comparison to:")
        print(cv_comparison_path)


if __name__ == "__main__":
    main()