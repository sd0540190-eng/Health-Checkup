# Health Checkup - Diabetes Risk Predictor

A machine learning web app that estimates a person's diabetes risk from 8 health values.

## How it works

Website (Frontend) -> FastAPI (Backend) -> Trained ML model (ML)

## Dataset

Pima Indians Diabetes dataset from Kaggle (768 rows, 8 features).
Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as missing values and filled with the median.

## Models

Logistic Regression, Random Forest and XGBoost are compared with 5-fold cross-validation. The best one (by ROC-AUC) is saved.

## Results

- Best model: (write your model name here)
- Test ROC-AUC: (write your score here)
- Test recall (diabetic): (write your score here)

## Project structure

```
health-checkup/
├── ML/          data, models, train.py, predict.py
├── Backend/     main.py (FastAPI)
├── Frontend/    index.html, style.css, script.js
└── requirements.txt
```

## How to run

1. Install packages:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Train the model again:
   ```
   cd ML
   python train.py
   ```
3. Start the API:
   ```
   cd Backend
   python -m uvicorn main:app --reload
   ```
4. Open `Frontend/index.html` in your browser.

## Disclaimer

This is a screening aid for learning purposes, not a medical diagnosis. Please consult a doctor for any health concern.