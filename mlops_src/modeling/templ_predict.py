from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import sklearn
import pandas as pd
import joblib
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from mlops_src.templ_config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = DATA_DIR / "external/X_test.csv",
    model_path: Path = MODELS_DIR / "lead_model_lr.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    X_test = pd.read_csv(features_path)
    predictions = model.predict(X_test)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(predictions_path, index=False)
    
    # -----------------------------------------


if __name__ == "__main__":
    app()


# OLD CODE from original source file (model_inference.py)
"""
with open("artifacts/lead_model_lr.pkl", "rb") as f:
    model = joblib.load(f)

X = pd.read_csv("artifacts/X_test.csv")
y = pd.read_csv("artifacts/y_test.csv")
print(model.predict(X.head(5)), y.head(5))
"""
