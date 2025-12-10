import pandas as pd
import joblib
import pytest
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from mlops_src.templ_config import MODELS_DIR, DATA_DIR


def test_load_test_data():
    # Define paths
    X_test_path = DATA_DIR / "external" / "X_test.csv"
    y_test_path = DATA_DIR / "external" / "y_test.csv"

    # These should ALWAYS exist as they are in Git
    assert X_test_path.exists(), f"X_test not found at {X_test_path}"
    assert y_test_path.exists(), f"y_test not found at {y_test_path}"

    # Load data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    assert not X_test.empty, "X_test is empty"
    assert not y_test.empty, "y_test is empty"
    assert len(X_test) == len(y_test), "Mismatch in X_test and y_test length"

def test_model_inference():
    # Define paths
    model_path = MODELS_DIR / "lead_model_lr.pkl"
    test_data_path = DATA_DIR / "external" / "X_test.csv"

    # Ensure files exist
    assert model_path.exists(), f"Model not found at {model_path}"
    assert test_data_path.exists(), f"Test data not found at {test_data_path}"

    # Load model
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    # Load test data (take a small sample)
    X_test = pd.read_csv(test_data_path).head(5)

    # Run inference
    predictions = model.predict(X_test)

    # Assertions
    assert len(predictions) == 5, "Prediction count mismatch"
    assert predictions is not None, "Predictions should not be None"

def main():
    test_load_test_data()
    test_model_inference()
    print("All inference tests passed.")
if __name__ == "__main__":
    main()
    