import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained model (ensure model.pkl exists in your working directory)
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define function for prediction
def predict_heart_disease(inputs):
    prediction = model.predict([inputs])
    return prediction[0]

# UI Setup
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### Enter your details below:")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.radio("Sex", ['Male', 'Female'])
chest_pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", ['No', 'Yes'])
resting_ecg = st.selectbox("Resting ECG Results", ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.radio("Exercise-Induced Angina", ['No', 'Yes'])

# Convert inputs into the required format
sex = 1 if sex == 'Male' else 0
fasting_bs = 1 if fasting_bs == 'Yes' else 0
exercise_angina = 1 if exercise_angina == 'Yes' else 0

chest_pain_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
resting_ecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}

chest_pain = chest_pain_dict[chest_pain]
resting_ecg = resting_ecg_dict[resting_ecg]

# Prediction Button
if st.button("Predict Heart Disease Risk"):
    inputs = [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina]
    result = predict_heart_disease(inputs)
    
    if result == 1:
        st.error("⚠ High risk of heart disease! Consult a doctor.")
        st.markdown("### 🏥 Health Recommendations:")
        st.write("- Maintain a balanced diet (low sodium, high fiber).")
        st.write("- Exercise regularly (30 min/day).")
        st.write("- Manage stress and get regular health checkups.")
    else:
        st.success("✅ Low risk of heart disease. Keep up the healthy lifestyle!")
        st.markdown("### 💪 Wellness Tips:")
        st.write("- Continue eating healthy and staying active.")
        st.write("- Monitor your blood pressure and cholesterol.")
        st.write("- Get adequate sleep and stay hydrated.")
