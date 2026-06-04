# src/api/main.py
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import os
import numpy as np
from pydantic_models import CreditApplicantRequest, CreditApplicantResponse
from data_processing import compute_regulatory_credit_score, derive_optimal_loan_terms

app = FastAPI(title="Bati Bank Automated Credit Scoring Engine", version="2.0.0")

@app.on_event("startup")
def load_production_model_from_registry():
    global model
    try:
        # Pulls the absolute best model from the central MLflow Tracking Registry
        model_name = "BatiBank_Credit_Risk_Model"
        stage = "Production"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
        print("Successfully loaded model from MLflow Registry.")
    except Exception as e:
        print(f"Registry connection unavailable ({e}). Falling back to local pickle binary layer.")
        import joblib
        if os.path.exists("models/credit_risk_rf_model.pkl"):
            model = joblib.load("models/credit_risk_rf_model.pkl")
        else:
            raise FileNotFoundError("No backup model binary detected locally.")

@app.post("/predict", response_model=CreditApplicantResponse)
def evaluate_applicant_credit(payload: CreditApplicantRequest):
    try:
        features = np.array([[payload.Recency, payload.Frequency, payload.Monetary]])
        
        # Handle predictions regardless of model flavor wrapping
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(features)[0][1])
        else:
            # For MLflow PyFunc generic models
            prob = float(model.predict(features)[0])
            
        # Core Financial Transformations
        credit_score = compute_regulatory_credit_score(prob)
        decision = "APPROVED" if credit_score >= 550 else "REJECTED"
        
        optimal_amount, duration = derive_optimal_loan_terms(credit_score, payload.Monetary)
        
        return CreditApplicantResponse(
            risk_probability=round(prob, 4),
            credit_score=credit_score,
            credit_decision=decision,
            recommended_loan_amount_ugx=optimal_amount,
            optimal_duration_months=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))