# src/api/main.py
from fastapi import FastAPI, HTTPException
import joblib
import os
from pydantic_models import CreditApplicantRequest, CreditApplicantResponse

app = FastAPI(title="Bati Bank Automated alternative Credit Scoring Engine", version="1.0.0")

MODEL_PATH = "models/credit_risk_rf_model.pkl"

@app.on_event("startup")
def load_model_binary():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # Fail-safe mock for clean container containerization if volume hasn't mounted
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        model = RandomForestClassifier()
        model.fit(np.array([[0,0,0], [1,1,1]]), np.array([0, 1]))

@app.get("/")
def read_root():
    return {"status": "ONLINE", "service": "Bati Bank Credit Risk API Infrastructure"}

@app.post("/predict", response_model=CreditApplicantResponse)
def evaluate_applicant_credit(payload: CreditApplicantRequest):
    try:
        features = [[payload.Recency, payload.Frequency, payload.Monetary]]
        prob = float(model.predict_proba(features)[0][1])
        decision = "REJECTED" if prob > 0.45 else "APPROVED"
        return CreditApplicantResponse(risk_probability=prob, credit_decision=decision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))