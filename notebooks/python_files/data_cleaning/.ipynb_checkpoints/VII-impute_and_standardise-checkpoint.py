from sklearn.preprocessing import MinMaxScaler
import joblib


def impute():
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("artifacts/cat_missing_impute.csv")
    
    # Continuous variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T
    
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T



def standardise():
    scaler_path = "./artifacts/scaler.pkl"
    
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    
    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in artifacts")
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    return cont_vars