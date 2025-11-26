import numpy as np
from pprint import pprint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from python_files.helper_functions import impute_missing_values, describe_numeric_col
import joblib
import json

def data_clean(data):
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    
    data = data[data.source == "signup"]
    result=data.lead_indicator.value_counts(normalize = True)
    
    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)
    return data

def create_separate_var(data):
    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]
    
    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")
    
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    print("\nContinuous columns: \n")
    pprint(list(cont_vars.columns), indent=4)
    print("\n Categorical columns: \n")
    pprint(list(cat_vars.columns), indent=4)

    return data, cat_vars, cont_vars


def outliers(data, cont_vars, out_summ_loc):
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                 upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(out_summ_loc)
    print(outlier_summary)


def impute(cat_vars, cont_vars):
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("artifacts/cat_missing_impute.csv")
    
    # Continuous variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
    
    return cat_vars, cont_vars



def standardise(cont_vars):
    scaler_path = "./artifacts/scaler.pkl"
    
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    
    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in artifacts")
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    return cont_vars


def combine(cont_vars, cat_vars):
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined.\nRows: {len(data)}")
    
    #THIS LINE COMES FROM THE CELL "DATA DRIFT ARTIFACT" IN THE OG NOTEBOOK
    data.to_csv('./artifacts/training_data.csv', index=False)


def drift_artifact(data, artifact_drift_path):
    data_columns = list(data.columns)
    with open(artifact_drift_path,'w+') as f:
        json.dump(data_columns,f)


def binning_and_saving(data, gold_data_path):
    data.columns

    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
               'fb' : 'socials', 
               'organic': 'group1', 
               'signup': 'group1'
               }
    
    data['bin_source'] = data['source'].map(mapping)
    
    
    #THIS LINE TAKEN FROM "SAVE GOLD MEDALLION DATASET" IN THE OG NOTEBOOK
    data.to_csv(gold_data_path, index=False)