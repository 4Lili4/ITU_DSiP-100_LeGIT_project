import pandas as pd
import datetime
import json

#!dvc pull

def load_train_data(raw_data_path):
    print("Loading training data")
    
    data = pd.read_csv(raw_data_path)
    
    print("Total rows:", data.count())
    display(data.head(5))

# testing
#max_date = "2024-01-31"
#min_date = "2024-01-01"
#if not max_date:
#    max_date = #pd.to_datetime(datetime.datetime.now().date()).date()
#else:
    
    max_date = pd.to_datetime("2024-01-31").date()
    min_date = pd.to_datetime("2024-01-01").date()

    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    return data, min_date, max_date

def feat_select(data):
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
        ],
        axis=1
    )
    
    #Removing columns that will be added back after the EDA
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )
    return data