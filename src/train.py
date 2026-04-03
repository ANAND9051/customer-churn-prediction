import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_models(data_path):
    print(f"--- Loading Processed Data: {data_path} ---")
    df = pd.read_csv(data_path)
    
    # 1. Split Features (X) and Target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 2. Train-Test Split (80% Train, 20% Test)
    # 'stratify=y' ensures both sets have the same % of churners.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. SMOTE (Only on Training Data to avoid Data Leakage!)
    # SMOTE creates synthetic examples of the minority class (Churners).
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Training set shape before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Training set shape after SMOTE: {y_train_res.value_counts().to_dict()}")
    
    # 4. Model 1: Logistic Regression (Baseline)
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_res, y_train_res)
    y_pred_lr = lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    
    # 5. Model 2: Random Forest (Powerful Ensemble)
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)
    
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")
    
    # 6. Save the Best Model and Feature Names
    # We save feature names so the Web App knows the exact order of columns.
    MODELS_DIR = "customer-churn-prediction/models"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(rf, f"{MODELS_DIR}/churn_model.pkl")
    joblib.dump(X.columns.tolist(), f"{MODELS_DIR}/feature_names.pkl")
    
    print(f"\n[Success] Model saved to {MODELS_DIR}/churn_model.pkl")

if __name__ == "__main__":
    DATA = "customer-churn-prediction/data/processed_churn_data.csv"
    if os.path.exists(DATA):
        train_models(DATA)
    else:
        print(f"Error: {DATA} not found. Run preprocess.py first.")
