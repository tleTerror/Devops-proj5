from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from typing import List

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the Pydantic model for the input data
class PredictRequest(BaseModel):
    data: List[List[float]]  # A list of lists, where each inner list is a list of floats

@app.get("/")
def read_root():
    return {"message": "This a minor project to learn devops!"}

@app.post("/predict")
def predict(request: PredictRequest):
    print("Received data:", request.data)  # Debug print
    data = np.array(request.data)
    prediction = model.predict(data)
    predicted_class = prediction.tolist()
    confidence = model.predict_proba(data)  # Get the confidence
    return {
        "prediction": predicted_class,
        "confidence": confidence.tolist()  # Include confidence score
    }

