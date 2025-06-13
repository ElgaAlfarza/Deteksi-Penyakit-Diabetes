import pickle
import joblib  # Perbaikan: menggunakan joblib untuk scaler
import streamlit as st
import pandas as pd

# Membaca Model dan Scaler
diabetes_model = pickle.load(open('Random Forest.sav', 'rb'))
scaler = joblib.load(open('scaler.sav', 'rb'))  # Pastikan 'scaler.sav' tersedia

# Judul Web
st.title(':)\nBy Elga Alfareza 2101010082')
st.title('Prediksi Diabetes: Deteksi dan Pencegahan Lebih Awal')

# Membuat Kolom
col1, col2 = st.columns(2)
with col1:
    gender = st.number_input('Input Nilai gender', format="%.0f")
with col2:    
    age = st.number_input('Input Nilai age', format="%.0f")
with col1:
    hypertension = st.number_input('Input Nilai hypertension', format="%.0f")
with col2:
    heart_disease = st.number_input('Input Nilai heart_disease', format="%.0f")
with col1:
    smoking_history = st.number_input('Input Nilai smoking_history', format="%.0f")
with col2:
    bmi = st.number_input('Input Nilai bmi', format="%.2f")
with col1:
    HbA1c_level = st.number_input('Input Nilai HbA1c_level', format="%.2f")
with col2:
    blood_glucose_level = st.number_input('Input Nilai blood_glucose_level', format="%.2f")

# Code untuk prediksi
diab_diagnosis = ''

# Tombol Prediksi
if st.button('Tes Prediksi Diabetes'):
    # Data input
    input_data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    }
    input_df = pd.DataFrame(input_data)
    
    # Normalisasi data input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    diab_prediction = diabetes_model.predict(input_scaled)

    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien Tidak Terkena Diabetes'
        
    st.success(diab_diagnosis)