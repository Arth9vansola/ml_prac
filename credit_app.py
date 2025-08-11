import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model, scaler, and columns
model = joblib.load('knn_credit.pkl')   
scaler = joblib.load('knn_credit_scaler.pkl')
feature_columns = joblib.load('knn_credit_columns.pkl')

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below and check if it's fraud or not.")

# Create input fields for each feature except Class
user_input = []
for col in feature_columns:
    value = st.number_input(f"Enter value for {col}", value=0.0, format="%.4f")
    user_input.append(value)

if st.button("Predict"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_columns)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")

st.caption("Model trained on Credit Card Fraud dataset.")
