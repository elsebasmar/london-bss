import pandas as pd
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from londonbss.ml_logic.registry import load_model

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(
    station_name: str
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    app.state.model = load_model(stage='production', n_station=str(station_name))

    model = app.state.model
    assert model is not None

    pred_es = model.predict(n=3984).to_json()

    return json.loads(pred_es)

@app.get("/")
def root():
    return {'greeting': 'Hello'}
