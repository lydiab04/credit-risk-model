## Production Setup & Installation

### Prerequisites
* Python 3.10+
* Docker & Docker Compose

### Local Environment Installation
1. Clone the repository and navigate to the root directory:
   ```bash
   git clone [https://github.com/lydiab04/credit-risk-model.git](https://github.com/lydiab04/credit-risk-model.git)
   cd credit-risk-model

## Credit Scoring Business Understanding

### 1. Basel II Regulatory Influence on Model Interpretability & Documentation
The Basel II Capital Accord heavily regulates how financial institutions manage credit risk, structural operational risks, and capital adequacy. Under Basel II’s **Internal Ratings-Based (IRB) approach**, banks are permitted to use internal empirical models to estimate key risk parameters: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). 

To prevent systemic risk and "black-box" manipulation, Basel II mandates rigorous **model governance, transparent auditing, and strict interpretability**. 
* **Auditability:** Internal or external regulators must be able to trace exactly why a specific applicant was denied credit or assigned a high risk score.
* **Explainability:** Traditional models (like Scorecards built on Weight of Evidence transformation and Logistic Regression) clearly quantify the marginal impact of each attribute. If a machine learning model is utilized, it must be accompanied by reliable explainability layers (e.g., SHAP, Lime, or structural constraints) to prevent disparate impact and meet compliance demands.
* **Capital Adequacy Linkage:** Because the estimated PD directly dictates how much risk-weighted capital Bati Bank must hold in reserve, uninterpretable or highly volatile models risk miscalculating reserves, leading to regulatory penalties or insolvency.

### 2. The Necessity of Proxy Variables and Inherent Business Risks
Because the Xente platform provides behavioral transaction histories but lacks explicit loan lifecycle histories (such as 30-, 60-, or 90-day past due historical records), a direct **Default Target Label ($Y$)** does not exist. To build a supervised machine learning model, we must engineer a **Proxy Variable** for credit risk based on **Recency, Frequency, and Monetary (RFM)** features. Customers showing low platform engagement, erratic spending patterns, or sudden drop-offs in transaction frequency will serve as a proxy for high-risk ("bad") borrowers.

#### Major Business Risks Implemented by Proxy-Based Predictions:
* **Target Misclassification (Label Noise):** A customer might stop transacting on the eCommerce platform simply because they switched to a competitor or closed their account, not because they are financially insolvent. Labeling them as "high risk" introduces noise.
* **Adverse Selection & False Positives:** The model may systematically deny credit to highly creditworthy individuals who happen to use the platform infrequently (High False Positives). This leads to lost loan origination revenue for Bati Bank.
* **Credit Contagion / Underestimated Risk (False Negatives):** High-frequency users with large transaction volumes might be flagged as low-risk ("good"). However, if their high volume is fueled by unsustainable personal leverage or fraudulent activities, the bank will mistakenly extend large credit lines, accelerating credit default losses.

### 3. Model Architecture Trade-Offs in Regulated Credit Contexts

| Evaluation Pillar | Simple/Interpretable Models (e.g., Logistic Regression + WoE) | High-Performance Models (e.g., XGBoost, LightGBM) |
| :--- | :--- | :--- |
| **Regulatory Compliance** | **Excellent.** Highly compliant. Linear coefficients and scorecard points map transparently to risk factors, meeting Basel II standards perfectly. | **Challenging.** Features undergo non-linear, high-degree interactions across thousands of decision splits, acting as a structural black-box. |
| **Feature Handling** | **Manual.** Requires tedious continuous-variable binning, monotonic constraints, and Weight of Evidence (WoE) calculations. | **Automated.** Naturally handles non-linear patterns, missing values, structural multicollinearity, and continuous variables without binning. |
| **Predictive Accuracy** | **Moderate.** Prone to underfitting if real-world relationships are deeply complex, multi-dimensional, or non-linear. | **High.** Maximizes the Area Under the ROC Curve (ROC-AUC) and captures nuanced alternative behavioral data signals effectively. |
| **Operational Stability** | **High.** Resistant to minor shifts in underlying data distributions; predictable behavior on extreme outliers. | **Volatile.** Prone to overfitting on niche dataset behaviors unless heavily regularized, which can lead to unexpected real-world drift. |