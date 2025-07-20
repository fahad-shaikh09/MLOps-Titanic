# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import train_model, predict
from prometheus_client import Counter, generate_latest
from fastapi.responses import Response

# Training model at startup
train_model()

# Define FastAPI app
app = FastAPI()

# Prometheus metrics
PREDICTION_COUNT = Counter("prediction_total", "Total predictions made")

# Request schema
class Passenger(BaseModel):
    Pclass: int
    Sex: int  # 0 = male, 1 = female
    Age: float
    Fare: float

@app.get("/")
def root():
    return {"message": "Titanic ML API is running!"}

@app.post("/predict")
def make_prediction(p: Passenger):
    PREDICTION_COUNT.inc()

    features = [p.Pclass, p.Sex, p.Age, p.Fare]
    result = predict(features)

    return {"survived": bool(result)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
