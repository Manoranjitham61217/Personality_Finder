import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Personality Prediction", layout="centered")
st.title("🧠 Personality Finder")

# ---------------- Load Files ----------------
@st.cache_resource
def load_files():
    model = joblib.load("personality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("Selected_features.pkl")
    input_cols = joblib.load("Input_values.pkl")   # list of feature names
    return model, scaler, selector, input_cols

model, scaler, selector, input_cols = load_files()

# Convert all column names to string (avoid numpy issues)
input_cols = [str(c) for c in input_cols]

st.subheader("📝 Adjust Your Personality Feature Levels")

# ---------------- Slider Inputs ----------------
input_values = {}

for col in input_cols:
    label = col.replace("_", " ").title()      # Clean name for UI
    input_values[col] = st.slider(
        label=label,
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

df_input = pd.DataFrame([input_values])

# ---------------- Prediction ----------------
if st.button("Predict Personality Type"):
    # Feature selection
    df_selected = selector.transform(df_input)

    # Scaling
    scaled_input = scaler.transform(df_selected)

    # Prediction
    pred = model.predict(scaled_input)[0]

    # Decode labels if encoder exists
    try:
        le = joblib.load("label_encoder.pkl")
        personality = le.inverse_transform([pred])[0]
    except:
       mapping = {
            0: "Introvert",
            1: "Extrovert",
            2: "Ambivert"
        }
       personality = mapping.get(pred, pred)

    st.success(f"🎯 Predicted Personality Type: **{personality}**")

    descriptions = {
        "Introvert": "You prefer calm environments and deep meaningful interactions.",
        "Extrovert": "You enjoy socializing, teamwork, and high-energy situations!",
        "Ambivert": "You balance introvert & extrovert traits effectively."
    }

    st.info(descriptions.get(personality))
