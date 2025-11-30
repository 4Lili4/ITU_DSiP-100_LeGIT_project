import pandas as pd
from config import interim_data, processed_data

data=pd.read_csv(interim_data)

#Binning the data
data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)

#Save gold medallion dataset
data.to_csv(processed_data, index=False)