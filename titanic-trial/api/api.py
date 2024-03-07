from fastapi import APIRouter, HTTPException
import pandas as pd
from fastapi.encoders import jsonable_encoder
from schemas import Health, Prediction, PredictionInput
from model.predict import predict
from model.train_pipeline import run_training

api_router = APIRouter()

@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict: 
    health = Health(version="0.0.1", name="Titanic API", health="OK", model_version="0.0.2")
    print(str(health))
    return health.dict()

@api_router.post("/predict", response_model=Prediction, status_code=200)
def predict_value(input_data: PredictionInput) -> Prediction: 
    # Convert data 
    print(str(input_data))
    input_df = pd.DataFrame(jsonable_encoder(input_data), index=range(1))
    print("Input df" + str(input_df))
    response = predict(input_data=input_df)
    return Prediction(value=True)
    
@api_router.put("/train", status_code=200)
def train_model() -> str: 
    response = run_training()
    return "Trained and persisted model"