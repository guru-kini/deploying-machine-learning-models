from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel) : 
    pclass: Optional[int]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]
    
    class Config: 
        schema_exta = {
            "example": {
                "pclass": 1,
                "sex": "male",
                "age": 30,
                "sibsp": 1,
                "parch": 2,
                "fare": 151.55,
                "cabin": "C22",
                "embarked": "S",
                "title": "Mr."
            }
        }
        