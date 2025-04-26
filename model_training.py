import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
import time

# Fungsi untuk mencetak log ke terminal
def log(message):
    print(message)

# Langkah 1: Memuat dataset
file_path = 'dataset.csv'  # Ganti dengan path dataset yang sesuai
log(f"Memuat dataset dari {file_path}...")

# Memuat data dalam format CSV
data = pd.read_csv(file_path)

# Menangani data kategorikal dengan One-Hot Encoding
log("Menangani data kategorikal dengan One-Hot Encoding...")
data_encoded = pd.get_dummies(data, drop_first=True)  # drop_first=True untuk menghindari multikolinearitas

log(f"Dataset setelah encoding:\n{data_encoded.head()}")

# Langkah 2: Menyiapkan Data
log("Mempersiapkan fitur dan target...")
X = data_encoded.drop(columns=['LotArea'])  # Ganti dengan nama kolom fitur yang sesuai
y = data['LotArea']  # Kolom target tidak terpengaruh oleh encoding

# Langkah 3: Membagi dataset menjadi training dan testing set
log("Membagi dataset menjadi training dan testing set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log("Data berhasil dibagi.")

# Menyimpan hasil waktu pelatihan untuk grafik
data_sizes = [1000, 2000, 3000, 4000, 5000]
bagging_times = []
xgboost_times = []

# Langkah 4: Melatih Model dengan Bagging dan XGBoost untuk ukuran data bertahap
for size in data_sizes:
    log(f"Melatih model dengan {size} data...")

    # Bagging (Random Forest) - Catat waktu pelatihan
    start_time = time.time()
    bagging_model = BaggingRegressor(n_estimators=10, random_state=42)
    bagging_model.fit(X_train[:size], y_train[:size])
    bagging_times.append(time.time() - start_time)

    # XGBoost - Catat waktu pelatihan
    start_time = time.time()
    xgboost_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgboost_model.fit(X_train[:size], y_train[:size])
    xgboost_times.append(time.time() - start_time)

    log(f"Model dengan {size} data berhasil dilatih.")

# Langkah 5: Grafik hubungan antara jumlah data dan waktu pelatihan
log("Membuat grafik hubungan antara jumlah data dan waktu pelatihan...")

plt.figure(figsize=(10, 6))
plt.plot(data_sizes, bagging_times, label='Bagging (Random Forest)', marker='o')
plt.plot(data_sizes, xgboost_times, label='XGBoost', marker='o')
plt.xlabel('Jumlah Data')
plt.ylabel('Waktu Pelatihan (detik)')
plt.title('Hubungan antara Jumlah Data dan Waktu Pelatihan')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Menyimpan grafik
plt.savefig('data_vs_training_time.png')
plt.show()

log("Grafik hubungan antara jumlah data dan waktu pelatihan berhasil dibuat.")

# Langkah 6: Melakukan prediksi dan evaluasi pada model akhir menggunakan data uji
log("Melakukan prediksi dengan model Bagging...")
bagging_pred = bagging_model.predict(X_test)

log("Melakukan prediksi dengan model XGBoost...")
xgboost_pred = xgboost_model.predict(X_test)

# Langkah 7: Menghitung MSE dan R² untuk Bagging
log("Menghitung MSE dan R² untuk model Bagging...")
bagging_mse = mean_squared_error(y_test, bagging_pred)
bagging_r2 = r2_score(y_test, bagging_pred)

log(f"Bagging Model MSE: {bagging_mse:.4f}, R²: {bagging_r2:.4f}")

# Menghitung MSE dan R² untuk XGBoost
log("Menghitung MSE dan R² untuk model XGBoost...")
xgboost_mse = mean_squared_error(y_test, xgboost_pred)
xgboost_r2 = r2_score(y_test, xgboost_pred)

log(f"XGBoost Model MSE: {xgboost_mse:.4f}, R²: {xgboost_r2:.4f}")

# Langkah 8: Menyimpan Model untuk Penggunaan Selanjutnya
log("Menyimpan model ke file...")
joblib.dump(bagging_model, 'bagging_model.pkl')
joblib.dump(xgboost_model, 'xgboost_model.pkl')
log("Model berhasil disimpan sebagai 'bagging_model.pkl' dan 'xgboost_model.pkl'.")

# Langkah 9: Membuat Grafik Prediksi Bagging vs XGBoost dalam 5 Grafik Terpisah
log("Membuat grafik prediksi Bagging dan XGBoost dalam grafik linear dan eksponensial...")

# Data untuk grafik
minutes = np.arange(1, 11)  # 10 menit pertama
data_count_linear = minutes * 1000  # Linear growth: 1000 data per menit
data_count_exp = 1000 * np.exp(0.2 * minutes)  # Eksponensial growth

# Grafik 1: Prediksi Bagging untuk Linear Growth
plt.figure(figsize=(10, 6))
plt.plot(data_count_linear, bagging_pred[:10], 'o', color='b', label="Prediksi Bagging (Linear)")
linear_regressor = LinearRegression()
linear_regressor.fit(data_count_linear.reshape(-1, 1), bagging_pred[:10])  # Fitting line for Bagging
plt.plot(data_count_linear, linear_regressor.predict(data_count_linear.reshape(-1, 1)), color='b', linestyle='--')
plt.title('Prediksi Bagging (Linear Growth)')
plt.xlabel('Jumlah Data (Linear Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()
plt.savefig('bagging_linear_growth.png')
plt.close()

# Grafik 2: Prediksi XGBoost untuk Linear Growth
plt.figure(figsize=(10, 6))
plt.plot(data_count_linear, xgboost_pred[:10], 'o', color='r', label="Prediksi XGBoost (Linear)")
linear_regressor.fit(data_count_linear.reshape(-1, 1), xgboost_pred[:10])  # Fitting line for XGBoost
plt.plot(data_count_linear, linear_regressor.predict(data_count_linear.reshape(-1, 1)), color='r', linestyle='--')
plt.title('Prediksi XGBoost (Linear Growth)')
plt.xlabel('Jumlah Data (Linear Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()
plt.savefig('xgboost_linear_growth.png')
plt.close()

# Grafik 3: Prediksi Bagging untuk Exponential Growth
plt.figure(figsize=(10, 6))
plt.plot(data_count_exp, bagging_pred[:10], 'o', color='b', label="Prediksi Bagging (Exponential)")
p = Polynomial.fit(data_count_exp, bagging_pred[:10], 2)  # Polynomial regression for Bagging
plt.plot(data_count_exp, p(data_count_exp), color='b', linestyle='--')
plt.title('Prediksi Bagging (Exponential Growth)')
plt.xlabel('Jumlah Data (Exponential Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()
plt.savefig('bagging_exponential_growth.png')
plt.close()

# Grafik 4: Prediksi XGBoost untuk Exponential Growth
plt.figure(figsize=(10, 6))
plt.plot(data_count_exp, xgboost_pred[:10], 'o', color='r', label="Prediksi XGBoost (Exponential)")
p = Polynomial.fit(data_count_exp, xgboost_pred[:10], 2)  # Polynomial regression for XGBoost
plt.plot(data_count_exp, p(data_count_exp), color='r', linestyle='--')
plt.title('Prediksi XGBoost (Exponential Growth)')
plt.xlabel('Jumlah Data (Exponential Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()
plt.savefig('xgboost_exponential_growth.png')
plt.close()

# Grafik 5: Perbandingan Prediksi Bagging vs XGBoost
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(data_count_linear, bagging_pred[:10], 'o', color='b', label="Prediksi Bagging (Linear)")
plt.plot(data_count_linear, xgboost_pred[:10], 'o', color='r', label="Prediksi XGBoost (Linear)")
linear_regressor.fit(data_count_linear.reshape(-1, 1), bagging_pred[:10])
plt.plot(data_count_linear, linear_regressor.predict(data_count_linear.reshape(-1, 1)), color='b', linestyle='--')
linear_regressor.fit(data_count_linear.reshape(-1, 1), xgboost_pred[:10])
plt.plot(data_count_linear, linear_regressor.predict(data_count_linear.reshape(-1, 1)), color='r', linestyle='--')
plt.title('Prediksi Bagging vs XGBoost (Linear Growth)')
plt.xlabel('Jumlah Data (Linear Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data_count_exp, bagging_pred[:10], 'o', color='b', label="Prediksi Bagging (Exponential)")
plt.plot(data_count_exp, xgboost_pred[:10], 'o', color='r', label="Prediksi XGBoost (Exponential)")
p = Polynomial.fit(data_count_exp, bagging_pred[:10], 2)
plt.plot(data_count_exp, p(data_count_exp), color='b', linestyle='--')
p = Polynomial.fit(data_count_exp, xgboost_pred[:10], 2)
plt.plot(data_count_exp, p(data_count_exp), color='r', linestyle='--')
plt.title('Prediksi Bagging vs XGBoost (Exponential Growth)')
plt.xlabel('Jumlah Data (Exponential Growth)')
plt.ylabel('Prediksi')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
