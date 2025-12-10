import pandas as pd
import joblib
import pytest
import os, sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from mlops_src.templ_config import MODELS_DIR, DATA_DIR
from mlops_src.templ_dataset import main as interim_processing
from mlops_src.templ_features import main as feature_processing
def check_test_dataset_exists():
    os.makedirs("tests/test_dataset", exist_ok=True)
    os.makedirs("tests/test_dataset/processed", exist_ok=True)
    os.makedirs("tests/test_dataset/raw", exist_ok=True)
    os.makedirs("tests/test_dataset/interim", exist_ok=True)

def test_load_training_data():
    # Define path
    raw_data_path = DATA_DIR / "raw" / "raw_data.csv"
    assert raw_data_path.exists(), f"Raw data not found at {raw_data_path}"

    # Load data
    df = pd.read_csv(raw_data_path)
    assert not df.empty, "Raw data is empty"
    assert len(df.columns) > 0, "Raw data has no columns"

def test_data_interim_processing():
    df = pd.read_csv(DATA_DIR / "raw" / "raw_data.csv").head(10)
    df.to_csv("tests/test_dataset/raw/raw_data_sample.csv", index=False)
    
    #Setting paths to test interim processing
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    INTERIM_DATA_DIR = PROJ_ROOT / "tests/test_dataset/interim"

    # Simulate interim processing by creating interim artifacts
    interim_processing(
    "tests/test_dataset/raw/raw_data_sample.csv",
    INTERIM_DATA_DIR / "date_limits.json",
    INTERIM_DATA_DIR / "outlier_summary.csv",
    INTERIM_DATA_DIR / "cat_missing_impute.csv",
    INTERIM_DATA_DIR / "scaler.pkl",
    INTERIM_DATA_DIR / "columns_drift.json",
    INTERIM_DATA_DIR / "interim_data.csv",
    0,
    0
    )
    # Check if interim data file is created
    assert (INTERIM_DATA_DIR / "interim_data.csv").exists(), "Interim data file not created"

def test_feature_processing():
    INTERIM_DATA_DIR = Path(__file__).resolve().parents[1] / "tests/test_dataset/interim"
    processed_data_path = INTERIM_DATA_DIR / "interim_data.csv"
    assert processed_data_path.exists(), f"Processed data not found at {processed_data_path}"

    # Load processed data
    df = pd.read_csv(processed_data_path)
    assert not df.empty, "Processed data is empty"
    assert len(df.columns) > 0, "Processed data has no columns"

    feature_processing(
        interim_path=processed_data_path,
        output_path="tests/test_dataset/processed/processed_data.csv"
    )

    # Check if final processed data file is created
    assert (Path(__file__).resolve().parents[1] / "tests/test_dataset/processed/processed_data.csv").exists(), "Final processed data file not created"

def feature_dimension_check():
    processed_data_path = Path(__file__).resolve().parents[1] / "tests/test_dataset/processed/processed_data.csv"
    df = pd.read_csv(processed_data_path)
    expected_num_features = 11  # Numbers of expected features
    assert df.shape[1] == expected_num_features, f"Expected {expected_num_features} features, found {df.shape[1]}"

def main():
    check_test_dataset_exists()
    test_load_training_data()
    test_data_interim_processing()
    test_feature_processing()
    feature_dimension_check()
    print("All data processing tests passed.")
if __name__ == "__main__":
    main()

