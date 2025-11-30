from pathlib import Path
import datetime

PROJ_ROOT = Path(__file__).resolve().parents[1]

#Paths:
data_dir = PROJ_ROOT / "data"
raw_data= data_dir / "raw/raw_data.csv"
interim_data = data_dir / "interim/interim_data.csv"
processed_data= data_dir / "processed/processed_data.csv"

max_date = "2024-01-31"
min_date = "2024-01-01"

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

artifact_path = "model"
model_name = "lead_model"