import pickle
import joblib
import streamlit as st
import pandas as pd

# --- MEMBACA MODEL DAN SCALER ---
# Pastikan semua file ini ada di folder yang sama dengan file .py Anda
try:
    diabetes_model = pickle.load(open('Random Forest.sav', 'rb'))
    scaler = joblib.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Pastikan file 'Random Forest.sav' dan 'scaler.sav' ada di folder yang sama.")
    st.stop()

# --- JUDUL WEB ---
st.title('Prediksi Diabetes: Deteksi Dini, Pencegahan Lebih Baik')
st.markdown('Oleh: Elga Alfareza - 2101010082')
st.write('---')

# --- MAPPING UNTUK DATA KATEGORI ---
# Kamus ini mengubah pilihan pengguna (teks) menjadi angka yang dimengerti model
gender_mapping = {'Wanita': 0, 'Pria': 1, 'Lainnya': 2}
hypertension_mapping = {'Tidak': 0, 'Ya': 1}
heart_disease_mapping = {'Tidak': 0, 'Ya': 1}
# CATATAN: Sesuaikan mapping ini dengan yang Anda gunakan saat melatih model
smoking_history_mapping = {
    'Tidak Pernah': 4,
    'Dulu Pernah (Mantan)': 3,
    'Perokok Saat Ini': 1,
    'Pernah Merokok': 2,
    'Tidak Ada Info': 0,
    'Bukan Perokok Saat Ini': 5
}

# --- MENAMPILKAN INFORMASI NILAI NUMERIK (SESUAI PERMINTAAN) ---
with st.expander("ℹ️ Lihat Informasi Nilai Numerik untuk Setiap Kategori"):
    st.write("**Gender:**")
    st.json(gender_mapping)
    
    st.write("**Riwayat Merokok:**")
    st.json(smoking_history_mapping)
    
    st.write("**Hipertensi & Penyakit Jantung:**")
    st.json(hypertension_mapping)

# --- INPUT PENGGUNA DENGAN LAYOUT KOLOM ---
st.header('Masukkan Data Pasien')
col1, col2 = st.columns(2)

with col1:
    # Menggunakan selectbox untuk data kategori agar lebih mudah
    gender_option = st.selectbox('Jenis Kelamin', options=list(gender_mapping.keys()))
    age = st.number_input('Umur (Tahun)', min_value=0, max_value=120, value=30, step=1)
    hypertension_option = st.selectbox('Memiliki Hipertensi?', options=list(hypertension_mapping.keys()))
    heart_disease_option = st.selectbox('Memiliki Penyakit Jantung?', options=list(heart_disease_mapping.keys()))

with col2:
    smoking_history_option = st.selectbox('Riwayat Merokok', options=list(smoking_history_mapping.keys()))
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=10.0, max_value=100.0, value=25.0, format="%.2f")
    HbA1c_level = st.number_input('Kadar HbA1c', min_value=3.0, max_value=16.0, value=6.0, format="%.2f")
    blood_glucose_level = st.number_input('Kadar Glukosa Darah', min_value=50, max_value=400, value=120, step=1)

# --- TOMBOL DAN LOGIKA PREDIKSI ---
diab_diagnosis = ''

if st.button('Cek Prediksi Diabetes', type="primary"):
    # Mengubah input teks dari selectbox menjadi angka menggunakan mapping
    gender = gender_mapping[gender_option]
    hypertension = hypertension_mapping[hypertension_option]
    heart_disease = heart_disease_mapping[heart_disease_option]
    smoking_history = smoking_history_mapping[smoking_history_option]

    # Menyiapkan data untuk prediksi
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
    
    # Normalisasi data input menggunakan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_df)

    # Melakukan prediksi
    diab_prediction = diabetes_model.predict(input_scaled)

    # Menampilkan hasil
    st.write("---")
    st.header("Hasil Prediksi")
    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien Berisiko Terkena Diabetes'
        st.error(diab_diagnosis, icon="⚠️")
    else:
        diab_diagnosis = 'Pasien Tidak Berisiko Terkena Diabetes'
        st.success(diab_diagnosis, icon="✅")
