import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score

# 1. Load data
df = pd.read_csv("data/diabetes.csv")
print("Shape:", df.shape)
print(df["Outcome"].value_counts())

# 2. Zeros are impossible for these columns, so treat them as missing
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Count non-diabetic (neg) and diabetic (pos) people, used by XGBoost
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()


# 4. Candidate models: fill missing -> scale -> model
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    "XGBoost": make_pipeline(
        XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      scale_pos_weight=neg / pos,
                      eval_metric="logloss", random_state=42)),
}

# 5. Compare models with cross-validation
best_name, best_score = None, 0
for name, pipe in models.items():
    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"{name}: CV ROC-AUC = {score:.3f}")
    if score > best_score:
        best_name, best_score = name, score

print("\nBest model:", best_name)

# 6. Train the best model and test it
best_model = models[best_name]
best_model.fit(X_train, y_train)

pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print("Test Recall (diabetic):", round(recall_score(y_test, pred), 3))
print(classification_report(y_test, pred))

# Check how recall changes with different thresholds
print("Recall at different thresholds:")
for t in [0.3, 0.4, 0.5]:
    p = (proba >= t).astype(int)
    print(f"  threshold {t}: recall = {recall_score(y_test, p):.2f}")

# 7. Save the model
joblib.dump(best_model, "models/diabetes_model.joblib")
joblib.dump(list(X.columns), "models/feature_names.joblib")
print("\nSaved to models/diabetes_model.joblib")