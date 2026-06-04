# src/data_processing.py (Updates for Task 3 & Task 5 Scorecard metrics)
import numpy as np

def compute_regulatory_credit_score(probability: float) -> int:
    """
    Transforms a model probability of default into a traditional financial credit score
    scaled between 300 and 850, mimicking Weight of Evidence (WoE) log-odds scaling.
    """
    # Prevent mathematical log(0) errors defensively
    prob = max(min(probability, 0.9999), 0.0001)
    
    # Calculate log-odds (similar to WoE scorecard foundations)
    log_odds = np.log((1 - prob) / prob)
    
    # Standard scorecard scaling formula: Score = Baseline + (Factor * Log-Odds)
    # Scales roughly such that a 1:1 odds ratio equals a score of 600
    baseline_score = 600
    factor = 50
    
    credit_score = int(baseline_score + (factor * log_odds))
    return max(min(credit_score, 850), 300)

def derive_optimal_loan_terms(credit_score: int, customer_monetary_value: float):
    """
    Predicts and assigns the optimal credit line amount and repayment duration
    based strictly on the customer's credit score tier and historical monetary volume.
    """
    # Convert log-transformed monetary back to raw estimate value safely
    raw_monetary_estimate = np.expm1(customer_monetary_value)
    
    if credit_score >= 700:    # Prime Tier
        amount_multiplier = 1.5
        duration_months = 12
    elif credit_score >= 550:  # Near-Prime Tier
        amount_multiplier = 0.8
        duration_months = 6
    else:                      # Sub-Prime Tier
        amount_multiplier = 0.2
        duration_months = 3
        
    optimal_amount = round(raw_monetary_estimate * amount_multiplier, -2)
    # Ensure a basic micro-loan floor if historical volume was zero
    optimal_amount = max(optimal_amount, 5000.0) 
    
    return optimal_amount, duration_months