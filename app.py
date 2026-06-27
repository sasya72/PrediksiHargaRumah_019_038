import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dashboard Prediksi Rumah", layout="wide")

# 1. LOAD HASIL TRAINING
@st.cache_resource
def load_data():
    with open('performa_model.pkl', 'rb') as f: db = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f: rf = pickle.load(f)
    with open('model_xgb.pkl', 'rb') as f: xgb = pickle.load(f)
    return db, rf, xgb

db, rf, xgb = load_data()

# 2. MENU SIDEBAR
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Analisis Karakteristik Data", "🔮 Prediksi Harga", "📊 Performa Model"])

# PERBAIKAN LOGIKA IF - ELIF - ELSE
if menu == "📊 Analisis Karakteristik Data":
    st.title("📊 Karakteristik Data (Boxplot)")
    st.write("Analisis distribusi perbandingan data training vs testing.")
    
    df = pd.read_excel(r"C:\Users\USER\Downloads\EAS ML PREDIKSI HARGA RUMAH\data_rumah.xlsx")
    
    X = df[['LB', 'LT', 'KT', 'KM', 'GRS']]
    y = df['HARGA']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    df_train = X_train.copy(); df_train['Harga'] = y_train; df_train['Set'] = 'Training'
    df_test = X_test.copy(); df_test['Harga'] = y_test; df_test['Set'] = 'Testing'
    df_plot = pd.concat([df_train, df_test])
    
    cols = ['Harga', 'LB', 'LT', 'KT', 'KM', 'GRS']
    for col in cols:
        fig = px.box(df_plot, x="Set", y=col, color="Set", title=f"Distribusi {col}")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🔮 Prediksi Harga":
    st.title("🏠 Prediksi Harga Rumah")
    
    col1, col2 = st.columns(2)
    with col1:
        lb = st.number_input("Luas Bangunan (m2)", 20, 2000, 150)
        lt = st.number_input("Luas Tanah (m2)", 20, 2000, 130)
        kt = st.slider("Kamar Tidur", 1, 10, 3)
        km = st.slider("Kamar Mandi", 1, 10, 2)
        grs = st.slider("Garasi", 0, 5, 1)
        model_choice = st.selectbox("Pilih Model", ["Random Forest Regressor", "XGBoost Regressor"])
    
    with col2:
        if st.button("🚀 Prediksi"):
            input_df = pd.DataFrame([[lb, lt, kt, km, grs]], columns=['LB', 'LT', 'KT', 'KM', 'GRS'])
            model = rf if "Random Forest" in model_choice else xgb
            harga = np.expm1(model.predict(input_df)[0])
            st.metric("Estimasi Harga", f"Rp {harga:,.0f}".replace(",", "."))

else: # Ini untuk menu "📊 Performa Model"
    st.title("📊 Analisis & Performa Model")
    st.subheader("Tabel Perbandingan")
    st.table(pd.DataFrame(db['tabel_perbandingan']))
    
    st.subheader("Parameter Terbaik")
    col1, col2 = st.columns(2)
    with col1: st.write("RF Params:", db['best_params_rf'])
    with col2: st.write("XGB Params:", db['best_params_xgb'])
