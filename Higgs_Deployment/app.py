import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained artifacts
model = joblib.load("best_model.pkl")
selected_features = joblib.load("selected_features.pkl")

st.set_page_config(page_title="Higgs Boson Detector", layout="centered")

st.title("🔬 Higgs Boson Signal Detection")
st.write("Enter detector feature values to predict whether the event is a Higgs Boson signal.")

# -------------------------------
# Input Form
# -------------------------------
st.subheader("Input Features")

input_data = {}

for feature in selected_features:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        format="%.5f"
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ Higgs Boson Signal Detected")
    else:
        st.error("❌ Background Event")

    st.write(f"**Confidence:** {probability:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("ML-based Higgs Boson Detection | Mini Project")
