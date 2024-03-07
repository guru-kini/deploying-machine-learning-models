from pydantic import BaseModel

class Prediction(BaseModel): 
    value: bool