import pandas as pd
import pytest
import os, sys
from pathlib import Path
from sklearn.utils.validation import check_is_fitted
import xgboost

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from mlops_src.modeling.templ_train import main as train_model
from mlops_src.templ_config import experiment_name, data_version, experiment_name, artifact_path, model_name

# from mlops_src.modeling.train import train_model # We will implement this later
def make_dirs():
    os.makedirs("tests/test_models", exist_ok=True)
    assert os.path.exists("tests/test_models"), "Failed to create test models directory"

def test_load_training_data():
    # Define path
    processed_data_path = "tests/test_dataset/processed/processed_data.csv"
    # Load data
    df = pd.read_csv(processed_data_path)
    assert not df.empty, "Processed data is empty"
    expected_num_features = 11
    assert df.shape[1] == expected_num_features, f"Expected {expected_num_features} features, found {df.shape[1]}"

def test_train_models():
    # Define paths
    MODELS_DIR = project_root + "/tests/test_models"
    INTERIM_DATA_DIR = project_root + "tests/test_dataset/interim"

    train_model(
    processed_path = "tests/test_dataset/processed/processed_data.csv",
    xgboost_model_path = MODELS_DIR + "/lead_model_xgboost.json",
    lr_model_path = MODELS_DIR + "/lead_model_lr.pkl",
    columns_list_path = INTERIM_DATA_DIR + "/columns_list.json",
    model_results_path = MODELS_DIR + "/model_results.json",
    LR_artifacts = INTERIM_DATA_DIR + "/LR_artifacts",
    metadata_path = MODELS_DIR + "/mlruns",
    
    experiment_name = experiment_name,
    data_version = data_version,
    artifact_path = artifact_path,
    model_name = model_name
    )
    # Check if models are created
    assert (Path(MODELS_DIR) / "lead_model_xgboost.json").exists(), "XGBoost model not created"
    assert (Path(MODELS_DIR) / "lead_model_lr.pkl").exists(), "Logistic Regression model not created"
    
def test_if_fitted():
    MODELS_DIR = project_root + "/tests/test_models"
    lr_model_path = MODELS_DIR + "/lead_model_lr.pkl"
    xgb_model_path = MODELS_DIR + "/lead_model_xgboost.json"

    with pytest.raises(Exception):
        check_is_fitted(lr_model_path)

    xgb_model = xgboost.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    booster = xgb_model.get_booster()
    assert booster is not None, "XGBoost model is not fitted"

def main():
    make_dirs()
    test_load_training_data()
    test_train_models()
    test_if_fitted()

    print("All training tests passed.")

if __name__ == "__main__":
    main()