from mlops_src.templ_config import DATA_DIR
import pandas as pd
import pytest
# from mlops_src.modeling.train import train_model # We will implement this later

def test_load_training_data():
    # Define path
    raw_data_path = DATA_DIR / "raw" / "raw_data.csv"

    # Skip if file doesn't exist (e.g. in Test pipeline where we don't pull raw data)
    if not raw_data_path.exists():
        pytest.skip("Raw data not found (likely running in Test mode)")

    # Load data
    df = pd.read_csv(raw_data_path)
    assert not df.empty, "Raw data is empty"
    assert len(df.columns) > 0, "Raw data has no columns"

def test_model_is_trainable():
    # Define path
    raw_data_path = DATA_DIR / "raw" / "raw_data.csv"
    
    if not raw_data_path.exists():
        pytest.skip("Raw data not found, cannot test training")

    # Load a small subset of data for the smoke test
    df = pd.read_csv(raw_data_path).head(50)
    
    # TODO: Implement this once we migrate the training code
    # model = train_model(df)
    # assert model is not None
    # assert hasattr(model, "predict")
    # pytest.fail("Training code not yet implemented")
