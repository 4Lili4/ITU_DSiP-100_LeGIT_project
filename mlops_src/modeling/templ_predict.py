from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


# OLD CODE from original source file (model_inference.py)
"""
import sklearn
import pandas as pd
import joblib

with open("artifacts/lead_model_lr.pkl", "rb") as f:
    model = joblib.load(f)

X = pd.read_csv("artifacts/X_test.csv")
y = pd.read_csv("artifacts/y_test.csv")
print(model.predict(X.head(5)), y.head(5))
"""
