from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_src.templ_config import MODELS_DIR, PROJ_ROOT, PROCESSED_DATA_DIR

import sklearn
import pandas as pd
import joblib

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROJ_ROOT / "/external/test_features.csv",
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
