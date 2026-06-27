import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os

# ==========================================
# FUNGSI EVALUASI
# ==========================================
def evaluate_model_full(model, X_train, y_train, X_test, y_test, name):
    # Prediksi Data Training
    preds_train_log = model.predict(X_train)
    preds_train_aktual = np.expm1(preds_train_log)
    y_train_aktual = np.expm1(y_train)
    r2_train = r2_score(y_train_aktual, preds_train_aktual)
    
    # Prediksi Data Testing
    preds_test_log = model.predict(X_test)
    preds_test_aktual = np.expm1(preds_test_log)
    y_test_aktual = np.expm1(y_test)
    r2_test = r2_score(y_test_aktual, preds_test_aktual)
    
    # Metrik Error pada Data Testing
    mae = mean_absolute_error(y_test_aktual, preds_test_aktual)
    rmse = np.sqrt(mean_squared_error(y_test_aktual, preds_test_aktual))
    mape = mean_absolute_percentage_error(y_test_aktual, preds_test_aktual) * 100
    
    return {
        "Model": name,
        "R² Train": r2_train,
        "R² Test": r2_test,
        "MAE (Juta)": mae / 1_000_000,
        "RMSE (Juta)": rmse / 1_000_000,
        "MAPE (%)": mape
    }

# ==========================================
# BLOK PENGAMAN (WAJIB DI WINDOWS UNTUK N_JOBS=-1)
# ==========================================
if __name__ == '__main__':
    # Load data asli dari excel
    pd.read_excel("data_rumah.xlsx")

    # Define features (x) and target (y)
    x = df[['LB', 'LT', 'KT', 'KM', 'GRS']]

    # Menggunakan log1p agar target harga stabil 
    y = np.log1p(df['HARGA'])

    # 1. MEMBAGI DATA (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Bentuk Data - X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    # 2. KARAKTERISTIK DATA (VISUALISASI BOXPLOT)
    print("⏳ Membuat visualisasi Boxplot Karakteristik Data...")
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")

    harga_train = np.expm1(y_train)
    harga_test = np.expm1(y_test)

    fitur_plot = [
        (harga_train, harga_test, 'Harga Rumah', 'Harga'),
        (X_train['LB'], X_test['LB'], 'Luas Bangunan (m2)', 'Luas Bangunan'),
        (X_train['LT'], X_test['LT'], 'Luas Tanah (m2)', 'Luas Tanah'),
        (X_train['KT'], X_test['KT'], 'Jumlah Kamar Tidur', 'Kamar Tidur'),
        (X_train['KM'], X_test['KM'], 'Jumlah Kamar Mandi', 'Kamar Mandi'),
        (X_train['GRS'], X_test['GRS'], 'Jumlah Garasi', 'Garasi')
    ]

    for i, (train_data, test_data, ylabel, title) in enumerate(fitur_plot):
        plt.subplot(2, 3, i+1)
        plot_data = pd.DataFrame({
            ylabel: np.concatenate([train_data, test_data]),
            'Dataset': ['Training'] * len(train_data) + ['Testing'] * len(test_data)
        })
        sns.boxplot(x='Dataset', y=ylabel, data=plot_data, palette="Set2")
        plt.title(f'Boxplot {title}')

    plt.tight_layout()
    plt.savefig("boxplot_karakteristik_data.png")
    print("✅ Gambar Boxplot berhasil disimpan sebagai 'boxplot_karakteristik_data.png'\n")

    # 3. PEMODELAN RANDOM FOREST & TUNING
    print("⏳ Mulai Hyperparameter Tuning Random Forest...")
    start_time = time.time()
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=rf_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    best_rf_model = rf_grid.best_estimator_
    print(f"✅ Tuning RF Selesai ({time.time() - start_time:.2f} detik)")

    # 4. PEMODELAN XGBOOST & TUNING
    print("⏳ Mulai Hyperparameter Tuning XGBoost...")
    start_time = time.time()
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 9],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    xgb_grid = GridSearchCV(estimator=XGBRegressor(random_state=42), 
                            param_grid=xgb_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    xgb_grid.fit(X_train, y_train)
    best_xgb_model = xgb_grid.best_estimator_
    print(f"✅ Tuning XGB Selesai ({time.time() - start_time:.2f} detik)\n")

    # 5. EVALUASI COMPREHENSIVE
    hasil_evaluasi = []
    hasil_evaluasi.append(evaluate_model_full(best_rf_model, X_train, y_train, X_test, y_test, "Random Forest Regressor"))
    hasil_evaluasi.append(evaluate_model_full(best_xgb_model, X_train, y_train, X_test, y_test, "XGBoost Regressor"))
    df_perbandingan = pd.DataFrame(hasil_evaluasi)

    if df_perbandingan.loc[0, 'R² Test'] > df_perbandingan.loc[1, 'R² Test']:
        best_model_name = "Random Forest Regressor"
    else:
        best_model_name = "XGBoost Regressor"

    performa = {
        "Random Forest Regressor": {
            "r2_val": df_perbandingan.loc[0, 'R² Test'], 
            "r2": f"{df_perbandingan.loc[0, 'R² Test']*100:.2f}%", 
            "mae": f"Rp {df_perbandingan.loc[0, 'MAE (Juta)'] * 1_000_000:,.0f}".replace(",", ".")
        },
        "XGBoost Regressor": {
            "r2_val": df_perbandingan.loc[1, 'R² Test'], 
            "r2": f"{df_perbandingan.loc[1, 'R² Test']*100:.2f}%", 
            "mae": f"Rp {df_perbandingan.loc[1, 'MAE (Juta)'] * 1_000_000:,.0f}".replace(",", ".")
        },
        "best_model": best_model_name,
        "best_params_rf": rf_grid.best_params_,
        "best_params_xgb": xgb_grid.best_params_,
        "tabel_perbandingan": df_perbandingan.to_dict(orient="records")
    }

    # 6. EXPORT BINER (.PKL)
    with open('performa_model.pkl', 'wb') as f:
        pickle.dump(performa, f)
    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(best_rf_model, f)
    with open('model_xgb.pkl', 'wb') as f:
        pickle.dump(best_xgb_model, f)

    print("=== TABEL PERBANDINGAN MODEL ===")
    print(df_perbandingan.to_string(index=False, float_format="%.4f"))
    print("\n✨ [SUKSES] Semua file integrasi .pkl berhasil diperbarui!")
