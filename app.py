import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("model/simple_churn_model.pkl")
scaler = joblib.load("model/simple_scaler.pkl")

st.title("📉 Customer Churn Prediction")

# ----- Inputs using sliders -----
# Numeric sliders
monthly_charges = st.slider("Monthly Charges (₹)", 18, 2000, 499)
tenure = st.slider("Tenure (months)", 0, 72, 12)

# Dropdown / Radio for categorical
contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
senior = st.radio("Senior Citizen?", ["No","Yes"])
gender = st.radio("Gender", ["Male","Female"])
paperless = st.radio("Paperless Billing?", ["No","Yes"])
internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
online_sec = st.radio("Online Security?", ["No","Yes"])
tech_support = st.radio("Tech Support?", ["No","Yes"])
device_protect = st.radio("Device Protection?", ["No","Yes"])
streaming_tv = st.radio("Streaming TV?", ["No","Yes"])
streaming_movies = st.radio("Streaming Movies?", ["No","Yes"])


# ----- Predict button -----
if st.button("Predict"):
    # Scale input
    input_data = scaler.transform([[tenure, monthly_charges]])
    # Prediction
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to churn – Probability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Customer is likely to stay – Probability: {prob*100:.2f}%")
