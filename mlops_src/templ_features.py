from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
import os,sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from mlops_src.templ_config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

#app = typer.Typer()


#@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    interim_path: Path = INTERIM_DATA_DIR / "interim_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    # -----------------------------------------
):
    
    data=pd.read_csv(interim_path, keep_default_na=False, float_precision='round_trip')
    
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
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
    #app()
