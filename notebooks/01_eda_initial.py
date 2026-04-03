import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for visualizations
sns.set_theme(style="whitegrid")

# 1. Load Data
# This dataset is a classic Telco Churn CSV.
DATA_PATH = "customer-churn-prediction/data/raw_churn_data.csv"
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH)

# 2. Initial Data Inspection
# info() tells us the data types and if there are null values.
print("--- Dataset Overview ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Data Cleaning (Crucial Step: TotalCharges is often an object due to spaces)
# We use errors='coerce' to turn those empty spaces " " into NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Let's see how many missing values we created by fixing 'TotalCharges'
print(f"\nMissing values after numeric conversion:\n{df.isnull().sum()}")

# 4. Basic Descriptive Statistics
# describe() gives us the Mean, Median, Min, and Max for numerical columns.
print("\n--- Descriptive Statistics (Numerical) ---")
print(df.describe())

# 5. Visualize Target Variable (Churn)
# This bar chart shows the "Class Imbalance" we discussed.
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
# In a CLI environment, we save the plot to a file instead of showing it.
plt.savefig('customer-churn-prediction/notebooks/churn_distribution.png')
print("\n[Action] Saved churn distribution plot to 'customer-churn-prediction/notebooks/churn_distribution.png'")

# 6. Check for Imbalanced Classes (The "Accuracy Trap" check)
churn_counts = df['Churn'].value_counts(normalize=True) * 100
print(f"\n--- Churn Percentage ---\n{churn_counts}")

# 7. Identifying Business Drivers (Quick look)
# Let's see if 'Contract' type affects churn significantly.
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df, palette='magma')
plt.title('Churn by Contract Type')
plt.savefig('customer-churn-prediction/notebooks/churn_by_contract.png')
print("[Action] Saved churn by contract plot to 'customer-churn-prediction/notebooks/churn_by_contract.png'")
