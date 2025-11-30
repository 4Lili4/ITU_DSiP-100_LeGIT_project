from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import warnings
import joblib
import typer
import json
import os

from templ_config import max_date, min_date, INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "raw_data.csv",
    date_limits_path: Path = INTERIM_DATA_DIR / "date_limits.json",
    outlier_sum_path: Path = INTERIM_DATA_DIR / "outlier_summary.csv",
    cat_impute_path: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv",
    scaler_path: Path = INTERIM_DATA_DIR / "scaler.pkl",
    columns_drift_path: Path = INTERIM_DATA_DIR / "columns_drift.json",
    interim_path: Path = INTERIM_DATA_DIR / "interim_data.csv",
    max_date=max_date,
    min_date=min_date
    # ----------------------------------------------
):

    #create artifacts directory if it doesn't already exist, BUT we use the interim folder instead!
    #os.makedirs("../data/artifacts",exist_ok=True)
    
    #Filter out pandas deprecation warning
    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format',lambda x: "%.3f" % x)
    
    #Read in the raw data found in the raw data folder
    data = pd.read_csv(input_path)
    
    #Making str max and min dates into datetime. If no max date, it chooses today's date. It's superfluous so i've commented it out
    #if not max_date:
    #    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    #else:
    #    max_date = pd.to_datetime(max_date).date()
    
    #Actually assigning values to min and max date, but some of the following may also be superfluous
    max_date = pd.to_datetime(max_date).date()
    min_date = pd.to_datetime(min_date).date()
    
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    
    #Saving the max and min dates in the data as artifacts
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open(date_limits_path, "w") as f:
        json.dump(date_limits, f)
    
    
    #CATEGORISED AS FEATURE SELECTION IN THE NOTEBOOK______________________
    data = data.drop(["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
        axis=1)
    
    #Removing columns that will be added back after the EDA
    data = data.drop(["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1)
    #______________________________________________________________________
    
    #Data cleaning, removing rows with empty target variable
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    #Data cleaning, removing rows with other invalid column data
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    
    data = data[data.source == "signup"]
    result=data.lead_indicator.value_counts(normalize = True)
    
    #Create categorical data columns
    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars:
        data[col] = data[col].astype("object")
    
    #Separate categorical and continuous columns
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    #Filter out outliers and save them as artifacts
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                 upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(outlier_sum_path)
    
    #Impute missing data for categorical variables
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(cat_impute_path)
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
    
    #Impute missing data for continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T
    
    #Data standardisation
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    
    joblib.dump(value=scaler, filename=scaler_path)
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    
    #Re-combine categorical and continuous data columns
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    
    #Save data drift artifact 
    data_columns = list(data.columns)
    with open(columns_drift_path,'w+') as f:           
        json.dump(data_columns,f)
    
    #Save interim training data
    data.to_csv(interim_path, index=False)


if __name__ == "__main__":
    app()
