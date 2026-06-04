# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_processing import generate_rfm_proxy_targets
import joblib
import os

def run_model_training(data_path):
    print("Loading raw transactional assets...")
    df = pd.read_csv(data_path)
    
    print("Engineering alternative proxy behavioral labels...")
    processed_features = generate_rfm_proxy_targets(df)
    
    X = processed_features[['Recency', 'Frequency', 'Monetary']]
    y = processed_features['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_test_split=0.25, random_state=42, stratify=y
    )
    
    print("Training production Random Forest baseline model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1_Score": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }
    
    print("\n--- Model Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/credit_risk_rf_model.pkl')
    print("\nModel binary compiled successfully to 'models/' directory.")

if __name__ == "__main__":
    # Ensure this points cleanly to your raw dataset
    if os.path.exists('data/raw/training_data.csv'):
        run_model_training('data/raw/training_data.csv')
    else:
        print("Data asset not found locally in data/raw/. Run training manually.")