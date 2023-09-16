import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from londonbss.ml_logic.registry import get_local_model

app = FastAPI()

# Loading model on startup

# Model eagle_wharf_road__hoxton
app.state.predictd = get_local_model()

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
    number_hours: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    num = number_hours

    pred = app.state.predictd

    return {'predictions' : int(1)}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
