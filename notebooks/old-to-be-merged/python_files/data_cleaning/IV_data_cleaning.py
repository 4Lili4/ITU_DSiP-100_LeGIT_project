import numpy as np

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