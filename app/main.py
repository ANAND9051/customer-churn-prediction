import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Configuration & Loading ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

# Dynamically find the project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    return None, None

model, feature_names = load_artifacts()

# --- App UI ---
st.title("🛡️ Customer Churn Prediction System")
st.markdown("""
Predict if a customer is likely to leave your service. 
This tool helps businesses proactively identify 'at-risk' customers and offer retention incentives.
""")

if model is None:
    st.error("Model artifacts not found. Please run 'train.py' first.")
    # This section helps us find the path error on the server
    with st.expander("🛠️ Debug Information (Check this to fix the error)"):
        st.write(f"**Current Directory (Server):** {os.getcwd()}")
        st.write(f"**Main Script Path:** {os.path.abspath(__file__)}")
        st.write(f"**Calculated Project Root:** {BASE_DIR}")
        st.write(f"**Searching for Model at:** {MODEL_PATH}")
        st.write(f"**Does Model Path Exist?** {'✅ Yes' if os.path.exists(MODEL_PATH) else '❌ No'}")
        
        # Show all files so we can see where the 'models' folder actually is
        st.write("**Files found in Project Root:**")
        if os.path.exists(BASE_DIR):
            st.write(os.listdir(BASE_DIR))
        else:
            st.write("Project Root directory not found!")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Customer Details")

# 1. Numerical Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 50)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)

# 2. Categorical Inputs
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])

# --- Preprocessing for Prediction ---
def prepare_input():
    # Create a base DataFrame with all zeros
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = 0.0
    
    # Map numerical values (Not scaled here for simplicity, but in a real app, use the saved scaler!)
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges
    
    # Map binary/one-hot columns (Only the ones we included in UI)
    if f"Contract_{contract}" in feature_names:
        input_df[f"Contract_{contract}"] = 1.0
    
    if f"InternetService_{internet}" in feature_names:
        input_df[f"InternetService_{internet}"] = 1.0
        
    if f"PaymentMethod_{payment}" in feature_names:
        input_df[f"PaymentMethod_{payment}"] = 1.0
        
    if f"TechSupport_Yes" in feature_names and tech_support == "Yes":
        input_df["TechSupport_Yes"] = 1.0
        
    if f"OnlineSecurity_Yes" in feature_names and online_security == "Yes":
        input_df["OnlineSecurity_Yes"] = 1.0
        
    return input_df

# --- Prediction Button ---
if st.button("🔍 Predict Churn Risk"):
    input_data = prepare_input()
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!**")
        st.write(f"The model predicts a **{probability*100:.1f}%** chance of this customer leaving.")
        st.progress(probability)
    else:
        st.success(f"✅ **Low Churn Risk**")
        st.write(f"The model predicts only a **{probability*100:.1f}%** chance of this customer leaving.")
        st.progress(probability)

    st.info("💡 **Recommendation:** Consider offering a long-term contract or tech support incentives.")

# --- Bottom Info ---
st.markdown("---")
st.caption("Built with Scikit-Learn, Pandas, and Streamlit. (Portfolio Project)")
