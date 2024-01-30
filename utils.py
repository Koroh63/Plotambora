DATASET_LOCATION = "CSV/1900_2021_DISASTERS.csv"

import pandas as pd

def importDataSet():
    return pd.read_csv(DATASET_LOCATION) 
    