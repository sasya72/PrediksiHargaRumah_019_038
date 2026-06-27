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
    
    @st.cache_data
    def load_data():
    # Pastikan hasil read_excel dimasukkan ke dalam variabel 'df'
    df_loaded = pd.read_excel("data_rumah.xlsx")
    return df_loaded

    # Panggil fungsi agar variabel 'df' tercipta di memori
    df = load_data()

    # Sekarang variabel df sudah bisa digunakan tanpa menyebabkan NameError
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
    
    # 1. Tabel Perbandingan
    st.subheader("Tabel Perbandingan Metrik Evaluasi")
    # Mengonversi list dictionary dari pickle menjadi DataFrame
    df_performa = pd.DataFrame(db['tabel_perbandingan'])
    
    # Menampilkan tabel dengan format yang lebih cantik
    st.dataframe(
        df_performa.style.format({
            'R² Train': '{:.4f}',
            'R² Test': '{:.4f}',
            'MAE (Juta)': 'Rp {:.2f} Juta',
            'RMSE (Juta)': 'Rp {:.2f} Juta',
            'MAPE (%)': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    st.divider() # Garis pemisah agar rapi
    
    # 2. Parameter Terbaik hasil Tuning
    st.subheader("Konfigurasi Hyperparameter Terbaik (Hasil Tuning)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌲 Random Forest Regressor**")
        st.json(db['best_params_rf'])
        st.caption(f"Skor Cross-Validation R²: {db['Random Forest Regressor']['cv_r2']:.4f}")
        
    with col2:
        st.markdown("**🚀 XGBoost Regressor**")
        st.json(db['best_params_xgb'])
        st.caption(f"Skor Cross-Validation R²: {db['XGBoost Regressor']['cv_r2']:.4f}")

    st.info(f"🏆 Kesimpulan: Berdasarkan hasil evaluasi, model **{db['best_model']}** adalah model dengan performa paling optimal untuk dataset ini.")
