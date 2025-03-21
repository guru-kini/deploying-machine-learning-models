import pandas as pd
import numpy as np
import re

from pathlib import Path
from model.config.configuration import DATASET_DIR

def get_first_cabin(row): 
    try:
        return row.split()[0]
    except:
        return np.nan
    
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
        
def load_data(*, file_name: str) -> pd.DataFrame : 
    print("Loading data...")
    # data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    print("...Loaded data. Massaging...")
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')
    
    # Drop unnecessary variables
    data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)
    
    print("...Data is ready to be used.")
    return data