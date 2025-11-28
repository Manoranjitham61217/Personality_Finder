import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Personality Prediction", layout="centered")
st.title("🧠 Personality Finder")

@st.cache_resource
def load_files():
    model = joblib.load("personality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("Selected_features.pkl")  # optional if required
    selected_features = joblib.load("selected_feature_names.pkl")  # <-- 🔥 only 10 features
    return model, scaler, selector, selected_features

model, scaler, selector, selected_features = load_files()

st.subheader("📝 Adjust Your Personality Feature Levels")

# -------- Build inputs only for selected 10 features -------- #
input_values = {}

for col in selected_features:
    input_values[col] = st.slider(
        col.replace("_", " ").title(),
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

# Create dataframe with only selected features
df_input = pd.DataFrame([input_values])

if st.button("Predict Personality Type"):
    
    # Scale only those selected features
    scaled_input = scaler.transform(df_input)

    # Predict
    pred = model.predict(scaled_input)[0]

    # Decode predicted class
    try:
        le = joblib.load("label_encoder.pkl")
        personality = le.inverse_transform([pred])[0]
    except:
        mapping = {0: "Introvert", 1: "Extrovert", 2: "Ambivert"}
        personality = mapping.get(pred, pred)

    st.success(f"🎯 Predicted Personality Type: **{personality}**")

    descriptions = {
        "Introvert": "You prefer calm environments and deep meaningful interactions.",
        "Extrovert": "You enjoy socializing, teamwork, and high-energy situations!",
        "Ambivert": "You balance both introversion and extroversion depending on the situation."
    }

    st.info(descriptions.get(personality))
