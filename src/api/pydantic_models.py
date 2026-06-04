# src/api/pydantic_models.py
from pydantic import BaseModel, Field

class CreditApplicantRequest(BaseModel):
    Recency: float = Field(..., example=4.0)
    Frequency: float = Field(..., example=1.38)
    Monetary: float = Field(..., example=6.81)

class CreditApplicantResponse(BaseModel):
    risk_probability: float
    credit_score: int = Field(..., description="Calculated score between 300-850")
    credit_decision: str
    recommended_loan_amount_ugx: float
    optimal_duration_months: int