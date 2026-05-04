from pathlib import Path
import pandas as pd


def load_data(filename="diabetic_data.csv"):
    """
    Load the diabetes readmission dataset.

    Looks for the file in common project data locations and raises a clear
    error if the file is missing.
    """
    project_root = Path(__file__).resolve().parent.parent

    candidate_paths = [
        project_root / "data" / "raw" / filename,
        project_root / "data" / filename,
        project_root / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            print(f"Loading data from: {path}")
            return pd.read_csv(path)

    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked these locations:\n{searched}\n\n"
        "Please place the dataset in one of those locations."
    )