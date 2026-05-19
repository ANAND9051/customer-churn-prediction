import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration & Loading ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

# Dynamically find the project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_1M_churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names_1M.pkl")


@st.cache_resource
def load_artifacts():
    # Check if both files exist
    model_exists = os.path.exists(MODEL_PATH)
    features_exists = os.path.exists(FEATURES_PATH)

    if not model_exists or not features_exists:
        return None, None, model_exists, features_exists

    try:
        model = joblib.load(MODEL_PATH)
        
        # --- GPU TO CPU FIX ---
        # If the model was saved with a GPU device, we force it to CPU for the server
        if hasattr(model, "set_params"):
            try:
                model.set_params(device="cpu")
            except:
                pass
        
        features = joblib.load(FEATURES_PATH)
        return model, features, True, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, True, True


model, feature_names, m_ok, f_ok = load_artifacts()

# --- App UI ---
st.title("🛡️ Enterprise Churn Predictor (1M Scale)")
st.markdown("""
Predict if a customer is likely to leave using our **89.56% Accuracy XGBoost Model**.
""")

if model is None:
    if not m_ok:
        st.error(f"❌ **Error:** Model file not found at `{MODEL_PATH}`")
    if not f_ok:
        st.error(f"❌ **Error:** Feature names file not found at `{FEATURES_PATH}`")
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
    # Create a base DataFrame with ALL ZEROS for every feature the model knows
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = 0.0

    # 1. Fill Numerical columns (If they exist in the model)
    if "tenure" in input_df.columns: input_df["tenure"] = float(tenure)
    if "MonthlyCharges" in input_df.columns: input_df["MonthlyCharges"] = float(monthly_charges)
    if "TotalCharges" in input_df.columns: input_df["TotalCharges"] = float(total_charges)

    # 2. Fill Categorical columns by matching the name exactly
    # We look through every column the model expects and set it to 1 if it matches our selection
    for col in input_df.columns:
        if f"Contract_{contract}" == col: input_df[col] = 1.0
        if f"InternetService_{internet}" == col: input_df[col] = 1.0
        if f"PaymentMethod_{payment}" == col: input_df[col] = 1.0
        if f"TechSupport_Yes" == col and tech_support == "Yes": input_df[col] = 1.0
        if f"OnlineSecurity_Yes" == col and online_security == "Yes": input_df[col] = 1.0

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
        st.write(
            f"The model predicts a **{probability * 100:.1f}%** chance of this customer leaving."
        )
        st.progress(probability)
    else:
        st.success(f"✅ **Low Churn Risk**")
        st.write(
            f"The model predicts only a **{probability * 100:.1f}%** chance of this customer leaving."
        )
        st.progress(probability)

    st.info(
        "💡 **Recommendation:** Consider offering a long-term contract or tech support incentives."
    )

# --- Bottom Info ---
st.markdown("---")
st.caption("Built with Scikit-Learn, Pandas, and Streamlit. (Portfolio Project)")
