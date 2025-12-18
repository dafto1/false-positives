from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the saved model
try:
    model = joblib.load("fraud_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    print("✅ Model loaded successfully.")
except:
    print("⚠️ Model files not found. Please run train_final.py first.")

class Transaction(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    category: str = "other"
    laundering_typology: str = "none"

@app.post("/predict")
def predict(tx: Transaction):
    # Convert input JSON to DataFrame
    data = pd.DataFrame([tx.dict()])
    
    # Align columns with training data (add missing, drop extra)
    for col in model_columns:
        if col not in data.columns:
            data[col] = 0
    data = data[model_columns]
    
    # Convert categoricals
    cat_cols = ["type", "category", "laundering_typology"]
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")
            
    # Make Prediction
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    
    return {
        "is_fraud": int(pred), 
        "risk_score": float(prob)
    }