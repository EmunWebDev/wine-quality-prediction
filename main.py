import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and imputer
model = joblib.load("models/wine_quality_model.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Page Config ---
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        .result-card {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 20px;
        }
        .good {
            background-color: #e0f7e9;
            color: #1b5e20;
            border: 2px solid #1b5e20;
        }
        .bad {
            background-color: #fdecea;
            color: #b71c1c;
            border: 2px solid #b71c1c;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🍷 Wine Quality Prediction App")
st.write("Enter wine chemical properties to predict if it's **Good** or **Not Good**.")

# --- Input Form ---
with st.form("wine_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)

    with col2:
        chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=0.1)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=0.1)
        density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")

    with col3:
        pH = st.number_input("pH", min_value=0.0, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("🔮 Predict")

# --- Prediction ---
if submitted:
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ])

    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    label = "🍷 Good Quality" if prediction == 1 else "❌ Not Good Quality"
    confidence = round(np.max(prediction_proba) * 100, 2)

    st.subheader("Prediction Result")
    st.markdown(
        f"""
        <div class="result-card {'good' if prediction == 1 else 'bad'}">
            <p>{label}</p>
            <p>Confidence: {confidence}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
