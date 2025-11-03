import datetime

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

data = pd.read_csv(data_gold_path)
print(f"Training data length: {len(data)}")
data.head(5)