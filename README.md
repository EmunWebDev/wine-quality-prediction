# 🍷 Wine Quality Prediction (Lab Exam)

This project predicts whether a red wine is **Good** (quality ≥ 7) or **Not Good** (quality < 7) based on chemical attributes.  
It was built using **Python, scikit-learn, joblib, and Streamlit**.

---

## 🚀 Live Demo
👉 [Try the App on Streamlit Cloud](https://wine-quality-prediction-emunwebdev.streamlit.app)

---

## 📂 Project Structure
labexam/
│── data/
│ └── winequality-red-selected-missing.csv # Dataset
│
│── models/
│ ├── train.py # Model training script
│ ├── wine_quality_model.pkl # Saved model
│ ├── imputer.pkl # Saved imputer for missing values
│ └── scaler.pkl # Saved scaler for feature scaling
│
│── main.py # Streamlit web app
│── requirements.txt # Project dependencies
