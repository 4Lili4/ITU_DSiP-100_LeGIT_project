from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier
import matplotlib.pyplot as plt
import mlflow.pyfunc
import joblib
import os

def XGmodel():
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

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def SKmodel():
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"
    
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
    
    
        # log artifacts
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")
        
        # store model for model interpretability
        joblib.dump(value=model, filename=lr_model_path)
            
        # Custom python model for predicting probability 
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))
    
    
    model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)
    
    best_model_lr_params = model_grid.best_params_