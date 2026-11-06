import joblib
import pandas as pd

model = joblib.load("models/diabetes_model.joblib")
features = joblib.load("models/feature_names.joblib")

def predict(data):
    df = pd.DataFrame([data])[features]
    prob = float(model.predict_proba(df)[0][1])
    return {"diabetes_risk": round(prob, 3), "prediction": int(prob >= 0.5)}

if __name__ == "__main__":
    sample = {"Pregnancies": 2, "Glucose": 140, "BloodPressure": 80,
              "SkinThickness": 30, "Insulin": 100, "BMI": 32.5,
              "DiabetesPedigreeFunction": 0.5, "Age": 45}
    print(predict(sample))