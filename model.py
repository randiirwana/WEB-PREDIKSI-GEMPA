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