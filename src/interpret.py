import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def interpret_model(model_path, data_path, features_path):
    print("--- Loading Model and Data for Interpretation ---")
    
    # 1. Load the artifacts
    model = joblib.load(model_path)
    # We only need the features to get the column names
    feature_names = joblib.load(features_path)
    
    # 2. Extract Feature Importance
    # Random Forest calculates 'Gini Importance' by default.
    # Higher values mean the feature was used more to split the trees.
    importances = model.feature_importances_
    
    # 3. Create a DataFrame for easy plotting
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 10 Drivers for Churn ---")
    print(feat_imp.head(10))
    
    # 4. Generate Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10), palette='viridis')
    plt.title("Top 10 Features Driving Churn (Random Forest Importance)")
    plt.xlabel("Importance Score")
    plt.ylabel("Customer Feature")
    
    # Save the plot
    PLOT_PATH = "customer-churn-prediction/notebooks/feature_importance.png"
    plt.savefig(PLOT_PATH, bbox_inches='tight')
    print(f"\n[Success] Feature Importance Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    MODEL = "customer-churn-prediction/models/churn_model.pkl"
    DATA = "customer-churn-prediction/data/processed_churn_data.csv"
    FEATURES = "customer-churn-prediction/models/feature_names.pkl"
    
    if os.path.exists(MODEL) and os.path.exists(FEATURES):
        interpret_model(MODEL, DATA, FEATURES)
    else:
        print("Error: Required files not found. Ensure you've run train.py.")
