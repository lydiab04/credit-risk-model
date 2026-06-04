# src/api/pydantic_models.py
from pydantic import BaseModel, Field

class CreditApplicantRequest(BaseModel):
    Recency: float = Field(..., description="Days since last active platform checkout entry")
    Frequency: float = Field(..., description="Log-transformed scale of total user occurrences")
    Monetary: float = Field(..., description="Log-scaled net volume expenditure footprint")

class CreditApplicantResponse(BaseModel):
    risk_probability: float = Field(..., description="Estimated statistical probability of credit default")
    credit_decision: str = Field(..., description="Systemic acceptance classification: APPROVED or REJECTED")