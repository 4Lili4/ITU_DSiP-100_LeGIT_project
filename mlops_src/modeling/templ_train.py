from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report#, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import mlflow.pyfunc
import mlflow
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier
from pathlib import Path
import pandas as pd
import joblib
import json
import time
import sys
import os
#import typer

# Get the project root (one directory above mlops_src)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from mlops_src.templ_config import PROCESSED_DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR, experiment_name, data_version, experiment_name, artifact_path, model_name
from mlops_src.deploy import deploy_to_staging

#app = typer.Typer()

def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"\nModel status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]



#@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    processed_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    xgboost_model_path: Path = MODELS_DIR / "lead_model_xgboost.json",
    lr_model_path: Path = MODELS_DIR / "lead_model_lr.pkl",
    columns_list_path: Path = INTERIM_DATA_DIR / "columns_list.json",
    model_results_path: Path = MODELS_DIR / "model_results.json",
    LR_artifacts: Path = INTERIM_DATA_DIR / "LR_artifacts",
    metadata_path: Path = MODELS_DIR / "mlruns",
    
    experiment_name = experiment_name,
    data_version = data_version,
    artifact_path = artifact_path,
    model_name = model_name
    # -----------------------------------------
):
    
    #Read in the processed training data
    data = pd.read_csv(processed_path)
    
    #Splitting the columns by variable-type
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols].copy()
    other_vars = data.drop(cat_cols, axis=1)
    
    #Dummy variable for categorical variables with one-hot encoding
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    #Re-combine variable-types into one dataframe again
    data = pd.concat([other_vars, cat_vars], axis=1)

    #Convert all columns to float
    for col in data:
        data[col] = data[col].astype("float64")
    
    #Splitting data into training and testing sets
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15, stratify=y)
    
    #TRAINING AND TESTING MODEL Random Forest
    model = XGBRFClassifier(random_state=42)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    
    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)
    model_grid.fit(X_train, y_train)
    best_model_xgboost_params = model_grid.best_params_
    
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)
    
    #Saving the best version of model Random forest
    xgboost_model = model_grid.best_estimator_
    xgboost_model.save_model(xgboost_model_path)

    xgboost_results = str(xgboost_model_path).split("\\")[-1]
    #Initialise dict to save model results, will later add results from logistic regression as well
    model_results = {xgboost_results: classification_report(y_train, y_pred_train, output_dict=True)}
    
    #Set mlflow model path to the models-folder, s.t. metadata created ends up in the mlruns-folder in models.
    mlflow.set_tracking_uri(metadata_path)
    
    #TRAINING AND TESTING MODEL Logistic regression
    mlflow.set_experiment(experiment_name)

    #Autolog is initialised, but cancelled out s.t. we only log the parameters we want (later)
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
    
        params = {
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  'penalty':  ["none", "l1", "l2", "elasticnet"],
                  'C' : [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)
    
        best_model = model_grid.best_estimator_
    
        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)
    
    
        # log artifacts - here we specify the parameters that we actually wish to log
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        # Ensure directory exists before logging
        os.makedirs(LR_artifacts, exist_ok=True)
        mlflow.log_artifacts(str(LR_artifacts), artifact_path="model")
        
        mlflow.log_param("data_version", data_version)
        
        # store model for model interpretability
        joblib.dump(value=best_model, filename=lr_model_path)
            
        # Custom python model for predicting probability 
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model))
    
    #Saving the best version of model Logistic Regression to the same dict we previously created
    model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)
    best_model_lr_params = model_grid.best_params_
    mlflow_results = str(lr_model_path).split("\\")[-1]
    model_results[mlflow_results] = model_classification_report
    
    #Save column-names and model-results as json-files
    with open(columns_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        json.dump(columns, columns_file)
    
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)
    
    #MODEL SELECTION______________________________________________________________________________________
    #Get experiment model results
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    
    with open(model_results_path, "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

    #Choose best model based on f1-score
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    
    #Get production model
    client = MlflowClient()
    prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
    prod_model_exists = len(prod_model)>0
    
    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']
        
        print('\n\nProduction model name: ', model_name)
        print('Production model version:', prod_model_version)
        print('Production model run id:', prod_model_run_id)
        
    else:
        print('\n\nNo model in production')
    
    #Compare prod and best trained model
    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None
    
    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]
    
        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score
    
        if train_model_score>prod_model_score:
            print("\nRegistering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]
    
    #Register best model
    if run_id is not None:
        print(f'\nBest model found: {run_id}')
    
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)
        print("Model details: ", model_details)

    #Transition to staging by calling function from other deploy.py-file in the mlops_src folder
    deploy_to_staging(model_name)


if __name__ == "__main__":
    main()
#    app()
