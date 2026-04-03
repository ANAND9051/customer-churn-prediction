import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data(input_path, output_path):
    print(f"--- Starting Preprocessing: {input_path} ---")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    
    # 2. Fix 'TotalCharges' (Convert to numeric and drop the 11 missing rows)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # 3. Drop 'customerID' (It's just an identifier, no predictive value)
    df.drop('customerID', axis=1, inplace=True)
    
    # 4. Binary Encoding (Yes/No -> 1/0)
    # We'll handle columns that only have two unique values first.
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Special case: gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # 5. One-Hot Encoding (Multi-category columns)
    # Columns like 'InternetService' have 3 options. 
    # pd.get_dummies creates a new column for each option.
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 6. Scaling (Tenure, MonthlyCharges, TotalCharges)
    # Why? Tenure is 1-72, MonthlyCharges is 20-120. Scaling puts them on the same level.
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 7. Save Processed Data
    df.to_csv(output_path, index=False)
    print(f"--- Preprocessing Complete! ---")
    print(f"New Shape: {df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    INPUT = "customer-churn-prediction/data/raw_churn_data.csv"
    OUTPUT = "customer-churn-prediction/data/processed_churn_data.csv"
    
    if os.path.exists(INPUT):
        preprocess_data(INPUT, OUTPUT)
    else:
        print(f"Error: {INPUT} not found.")
