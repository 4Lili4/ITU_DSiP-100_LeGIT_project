from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import mlflow.pyfunc
import mlflow
import json
import joblib
import os

def XGaccuracy():
    #Model test accuracy
    best_model_xgboost_params = model_grid.best_params_
    print("Best xgboost params")
    pprint(best_model_xgboost_params)
    
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)
    print("Accuracy train", accuracy_score(y_pred_train, y_train ))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))

def XGperf():
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test),'\n')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train),'\n')
    
    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)

def best_model():
    model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True) }

def SKparams():
    print("Best lr params")
    pprint(best_model_lr_params)
    
    print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
    print("Accuracy test:", accuracy_score(y_pred_test, y_test))
    
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test),'\n')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train),'\n')
    
    model_results[lr_model_path] = model_classification_report
    print(model_classification_report["weighted avg"]["f1-score"])

    model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)

def save_columns():
    column_list_path = './artifacts/columns_list.json'
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        pprint(columns)
        json.dump(columns, columns_file)
    
    print('Saved column list to ', column_list_path)

