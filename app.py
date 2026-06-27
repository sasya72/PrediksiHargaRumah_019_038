import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dashboard Prediksi Rumah", layout="wide")

# 1. LOAD MODEL (Fungsi Global)
@st.cache_resource
def load_models():
    # Menggunakan os.path agar lebih aman saat di-deploy
    cwd = os.getcwd()
    with open(os.path.join(cwd, 'performa_model.pkl'), 'rb') as f: db = pickle.load(f)
    with open(os.path.join(cwd, 'model_rf.pkl'), 'rb') as f: rf = pickle.load(f)
    with open(os.path.join(cwd, 'model_xgb.pkl'), 'rb') as f: xgb = pickle.load(f)
    return db, rf, xgb

db, rf, xgb = load_models()

# 2. MENU SIDEBAR
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Analisis Karakteristik Data", "🔮 Prediksi Harga", "📊 Performa Model"])

# 3. LOGIKA HALAMAN
if menu == "📊 Analisis Karakteristik Data":
    st.title("📊 Karakteristik Data (Boxplot)")
    st.write("Analisis distribusi perbandingan data training vs testing.")
    
    # Fungsi pembaca Excel terpisah agar tidak konflik
    @st.cache_data
    def load_excel_data():
        return pd.read_excel("data_rumah.xlsx")

    df = load_excel_data()

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
            # Prediksi dalam log, expm1 mengembalikannya ke nilai rupiah asli
            harga = np.expm1(model.predict(input_df)[0])
            st.metric("Estimasi Harga", f"Rp {harga:,.0f}".replace(",", "."))

else: # Menu "📊 Performa Model"
    st.title("📊 Analisis & Performa Model")
    
    st.subheader("Tabel Perbandingan Metrik Evaluasi")
    df_performa = pd.DataFrame(db['tabel_perbandingan'])
    st.dataframe(df_performa.style.format({
        'R² Train': '{:.4f}', 'R² Test': '{:.4f}',
        'MAE (Juta)': 'Rp {:.2f} Juta', 'RMSE (Juta)': 'Rp {:.2f} Juta', 'MAPE (%)': '{:.2f}%'
    }), use_container_width=True)
    
    st.divider()
    
    st.subheader("Konfigurasi Hyperparameter Terbaik")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌲 Random Forest Regressor**")
        st.json(db['best_params_rf'])
        st.caption(f"Skor Cross-Validation R²: {db['Random Forest Regressor']['cv_r2']:.4f}")
    with col2:
        st.markdown("**🚀 XGBoost Regressor**")
        st.json(db['best_params_xgb'])
        st.caption(f"Skor Cross-Validation R²: {db['XGBoost Regressor']['cv_r2']:.4f}")

    st.info(f"🏆 Kesimpulan: Model **{db['best_model']}** adalah model paling optimal.")
