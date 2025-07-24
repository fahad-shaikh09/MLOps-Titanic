# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import train_model, predict
from prometheus_client import Counter, generate_latest
from fastapi.responses import Response


# Add imports for whylogs
import pandas as pd
import whylogs as why
from datetime import datetime
import os

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
    
    # Prepare DataFrame for monitoring
    df = pd.DataFrame([{
        "Pclass": p.Pclass,
        "Sex": p.Sex,
        "Age": p.Age,
        "Fare": p.Fare,
        "Survived": int(result)
    }])

    # Log profile using whylogs
    profile = why.log(pandas=df).profile()

    # Save profile locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("whylogs_output", exist_ok=True)
    profile.write(f"whylogs_output/profile_{timestamp}.bin")
    return {"survived": bool(result)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
