import typing as t
import pandas as pd
from .persistence import persisted_model_path, load_pipeline

def load_model(): 
    file_name = persisted_model_path()
    return load_pipeline(file_name=file_name)
    
titanic_pipeline = load_model()    

def predict(*,input_data: t.Union[pd.DataFrame, dict],) -> dict:
    data = pd.DataFrame(input_data)
    
    if titanic_pipeline is None: 
        raise Exception("The pipeline is not trained and ready yet!")
    
    prediction = titanic_pipeline.predict(data)
    return prediction
    