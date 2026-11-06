from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Find the saved model (works on Windows, Mac and Linux)
MODEL_DIR = Path(__file__).resolve().parent.parent / "ML" / "models"
model = joblib.load(MODEL_DIR / "diabetes_model.joblib")
features = joblib.load(MODEL_DIR / "feature_names.joblib")

app = FastAPI(title="Health Checkup API")

# Allow the website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def home():
    return {"message": "Health Checkup API is running"}


@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])[features]

    # Same rule as training: 0 means "missing" for these columns
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, np.nan)

    prob = float(model.predict_proba(df)[0][1])
    return {
        "diabetes_risk": round(prob, 3),
        "prediction": int(prob >= 0.5),
        "disclaimer": "This is a screening aid, not a medical diagnosis.",
    }