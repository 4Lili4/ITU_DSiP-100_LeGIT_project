import os
import shutil
import mlflow

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)