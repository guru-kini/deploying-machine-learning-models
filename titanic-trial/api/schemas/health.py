from pydantic import BaseModel

class Health(BaseModel): 
    version: str
    name: str
    health: str
    model_version: str