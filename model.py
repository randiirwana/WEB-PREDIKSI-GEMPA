import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import joblib

# 1. Baca dataset dari data.csv
print("Membaca dataset...")
df = pd.read_csv('data.csv')

# 2. Pilih fitur dan label
X = df[['latitude', 'longitude', 'depth']]
y = df['magnitude']

# 3. Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Buat dan latih model Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
print("\nMelatih model...")
model.fit(X_train, y_train)

# 6. Prediksi data uji
y_pred = model.predict(X_test)

# 7. Evaluasi Model (MAE & RMSE)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nMAE:", mae)
print("RMSE:", rmse)

# Simpan MAE & RMSE ke file
with open('metrics.txt', 'w') as f:
    f.write(f"{mae}\n{rmse}\n")

# 8. Plot Feature Importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": ['latitude', 'longitude', 'depth'],
    "Importance": feature_importances
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
fig_fi = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
               title="Fitur Terpenting pada Random Forest",
               color="Importance", color_continuous_scale=px.colors.sequential.Viridis)
fig_fi.update_layout(
    yaxis_title="Fitur",
    xaxis_title="Tingkat Kepentingan",
    yaxis_autorange="reversed",
    template='plotly_white'
)
os.makedirs('static/plots', exist_ok=True)
fig_fi.write_html('static/plots/feature_importance.html', auto_open=False)

# 9. Simpan model dan scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nModel dan scaler berhasil disimpan.")

# --- MODEL TAHUNAN ---
# Hitung jumlah gempa per tahun
print("\nTraining model jumlah gempa tahunan...")
df['date'] = pd.to_datetime(df['date'])
df['tahun'] = df['date'].dt.year
df['bulan'] = df['date'].dt.month
df_tahun = df.groupby('tahun').size().reset_index(name='jumlah_gempa')
X_tahun = df_tahun[['tahun']]
y_tahun = df_tahun['jumlah_gempa']
X_train_tahun, X_test_tahun, y_train_tahun, y_test_tahun = train_test_split(X_tahun, y_tahun, test_size=0.2, random_state=42)
model_tahun = RandomForestRegressor(n_estimators=100, random_state=42)
model_tahun.fit(X_train_tahun, y_train_tahun)
y_pred_tahun = model_tahun.predict(X_test_tahun)
mae_tahun = mean_absolute_error(y_test_tahun, y_pred_tahun)
rmse_tahun = np.sqrt(mean_squared_error(y_test_tahun, y_pred_tahun))
print("MAE Tahunan:", mae_tahun)
print("RMSE Tahunan:", rmse_tahun)
joblib.dump(model_tahun, 'model_jumlah_gempa_tahunan.joblib')

# --- MODEL BULANAN ---
print("\nTraining model jumlah gempa bulanan...")
df_bulan = df.groupby(['tahun', 'bulan']).size().reset_index(name='jumlah_gempa')
X_bulan = df_bulan[['tahun', 'bulan']]
y_bulan = df_bulan['jumlah_gempa']
X_train_bulan, X_test_bulan, y_train_bulan, y_test_bulan = train_test_split(X_bulan, y_bulan, test_size=0.2, random_state=42)
model_bulan = RandomForestRegressor(n_estimators=100, random_state=42)
model_bulan.fit(X_train_bulan, y_train_bulan)
y_pred_bulan = model_bulan.predict(X_test_bulan)
mae_bulan = mean_absolute_error(y_test_bulan, y_pred_bulan)
rmse_bulan = np.sqrt(mean_squared_error(y_test_bulan, y_pred_bulan))
print("MAE Bulanan:", mae_bulan)
print("RMSE Bulanan:", rmse_bulan)
joblib.dump(model_bulan, 'model_jumlah_gempa_bulanan.joblib')

print("\nModel tahunan dan bulanan berhasil disimpan.")
# Setelah training model tahunan dan bulanan
with open('metrics_jumlah_gempa.txt', 'w') as f:
    f.write(f"{mae_tahun}\n{rmse_tahun}\n{mae_bulan}\n{rmse_bulan}\n")
print("\nMAE & RMSE model tahunan dan bulanan berhasil disimpan ke metrics_jumlah_gempa.txt")

# --- GRAFIK TAHUNAN ---
# Data asli tahunan
fig_tahun = go.Figure()
fig_tahun.add_trace(go.Bar(x=df_tahun['tahun'], y=df_tahun['jumlah_gempa'], name='Data Asli', marker_color='royalblue'))
# Prediksi model (training set)
y_pred_train_tahun = model_tahun.predict(X_tahun)
fig_tahun.add_trace(go.Bar(x=df_tahun['tahun'], y=y_pred_train_tahun, name='Prediksi Model', marker_color='orange'))
fig_tahun.update_layout(title='Jumlah Gempa per Tahun', xaxis_title='Tahun', yaxis_title='Jumlah Gempa', barmode='group')
fig_tahun.write_html('static/plots/grafik_tahunan.html', auto_open=False)

# --- GRAFIK BULANAN ---
# Data asli bulanan
label_bulan = df_bulan['tahun'].astype(str) + '-' + df_bulan['bulan'].astype(str).str.zfill(2)
fig_bulan = go.Figure()
fig_bulan.add_trace(go.Bar(x=label_bulan, y=df_bulan['jumlah_gempa'], name='Data Asli', marker_color='royalblue'))
y_pred_train_bulan = model_bulan.predict(X_bulan)
fig_bulan.add_trace(go.Bar(x=label_bulan, y=y_pred_train_bulan, name='Prediksi Model', marker_color='orange'))
fig_bulan.update_layout(title='Jumlah Gempa per Bulan', xaxis_title='Tahun-Bulan', yaxis_title='Jumlah Gempa', barmode='group', xaxis_tickangle=-45)
fig_bulan.write_html('static/plots/grafik_bulanan.html', auto_open=False)