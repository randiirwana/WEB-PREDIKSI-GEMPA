from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import logging
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisis', methods=['GET', 'POST'])
def analisis():
    tahun = request.form.get('tahun')
    bulan = request.form.get('bulan')
    mode_prediksi = request.form.get('mode_prediksi')
    hasil_prediksi = None
    grafik_prediksi_path = None

    df = pd.read_csv('data.csv')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['tahun'] = df['date'].dt.year
        df['bulan'] = df['date'].dt.month

    if tahun:
        df = df[df['tahun'] == int(tahun)]
    if bulan:
        df = df[df['bulan'] == int(bulan)]
    jumlah_gempa = len(df)

    # Proses prediksi jika ada input form
    if request.method == 'POST' and mode_prediksi:
        import plotly.graph_objects as go
        import joblib
        import numpy as np
        os.makedirs('static/plots', exist_ok=True)
        if mode_prediksi == 'tahunan' and tahun:
            model_tahun = joblib.load('model_jumlah_gempa_tahunan.joblib')
            X_pred = np.array([[int(tahun)]])
            hasil_prediksi = model_tahun.predict(X_pred)[0]
            # Buat grafik bar chart hasil prediksi
            fig = go.Figure([go.Bar(x=[str(tahun)], y=[hasil_prediksi], marker_color='royalblue')])
            fig.update_layout(title=f'Prediksi Jumlah Gempa Tahun {tahun}', xaxis_title='Tahun', yaxis_title='Jumlah Gempa')
            grafik_prediksi_path = f'static/plots/prediksi_tahun_{tahun}.html'
            fig.write_html(grafik_prediksi_path, auto_open=False)
            grafik_prediksi_path = grafik_prediksi_path.replace('static/', '')
        elif mode_prediksi == 'bulanan' and tahun and bulan:
            model_bulan = joblib.load('model_jumlah_gempa_bulanan.joblib')
            X_pred = np.array([[int(tahun), int(bulan)]])
            hasil_prediksi = model_bulan.predict(X_pred)[0]
            # Buat grafik bar chart hasil prediksi
            label_bulan = f'{tahun}-{bulan.zfill(2)}'
            fig = go.Figure([go.Bar(x=[label_bulan], y=[hasil_prediksi], marker_color='orange')])
            fig.update_layout(title=f'Prediksi Jumlah Gempa Bulan {label_bulan}', xaxis_title='Tahun-Bulan', yaxis_title='Jumlah Gempa')
            grafik_prediksi_path = f'static/plots/prediksi_bulan_{tahun}_{bulan}.html'
            fig.write_html(grafik_prediksi_path, auto_open=False)
            grafik_prediksi_path = grafik_prediksi_path.replace('static/', '')

    mae_tahun, rmse_tahun, mae_bulan, rmse_bulan = read_metrics_jumlah_gempa()
    return render_template(
        'analisis.html',
        jumlah_gempa=jumlah_gempa,
        tahun=tahun,
        bulan=bulan,
        mae_tahun=mae_tahun,
        rmse_tahun=rmse_tahun,
        mae_bulan=mae_bulan,
        rmse_bulan=rmse_bulan,
        hasil_prediksi=hasil_prediksi,
        grafik_prediksi_path=grafik_prediksi_path,
        mode_prediksi=mode_prediksi
    )

# Tambahkan fungsi untuk membaca hasil MAE dan RMSE dari file metrics_jumlah_gempa.txt dan kirim ke aplikasi.html

def read_metrics_jumlah_gempa():
    try:
        with open('metrics_jumlah_gempa.txt', 'r') as f:
            lines = f.readlines()
            mae_tahun = float(lines[0].strip())
            rmse_tahun = float(lines[1].strip())
            mae_bulan = float(lines[2].strip())
            rmse_bulan = float(lines[3].strip())
            return mae_tahun, rmse_tahun, mae_bulan, rmse_bulan
    except Exception:
        return None, None, None, None

@app.route('/aplikasi', methods=['GET', 'POST'])
def aplikasi():
    mode_prediksi = None
    tahun = None
    bulan = None
    hasil_prediksi = None
    grafik_prediksi_path = None

    if request.method == 'POST':
        mode_prediksi = request.form.get('mode_prediksi')
        tahun = request.form.get('tahun')
        bulan = request.form.get('bulan')
        import plotly.graph_objects as go
        import joblib
        import numpy as np
        import os
        os.makedirs('static/plots', exist_ok=True)
        if mode_prediksi == 'tahunan' and tahun:
            model_tahun = joblib.load('model_jumlah_gempa_tahunan.joblib')
            X_pred = np.array([[int(tahun)]])
            hasil_prediksi = model_tahun.predict(X_pred)[0]
            fig = go.Figure([go.Bar(x=[str(tahun)], y=[hasil_prediksi], marker_color='royalblue')])
            fig.update_layout(title=f'Prediksi Jumlah Gempa Tahun {tahun}', xaxis_title='Tahun', yaxis_title='Jumlah Gempa')
            grafik_prediksi_path = f'static/plots/prediksi_tahun_aplikasi_{tahun}.html'
            fig.write_html(grafik_prediksi_path, auto_open=False)
            grafik_prediksi_path = grafik_prediksi_path.replace('static/', '')
        elif mode_prediksi == 'bulanan' and tahun and bulan:
            model_bulan = joblib.load('model_jumlah_gempa_bulanan.joblib')
            X_pred = np.array([[int(tahun), int(bulan)]])
            hasil_prediksi = model_bulan.predict(X_pred)[0]
            label_bulan = f'{tahun}-{str(bulan).zfill(2)}'
            fig = go.Figure([go.Bar(x=[label_bulan], y=[hasil_prediksi], marker_color='orange')])
            fig.update_layout(title=f'Prediksi Jumlah Gempa Bulan {label_bulan}', xaxis_title='Tahun-Bulan', yaxis_title='Jumlah Gempa')
            grafik_prediksi_path = f'static/plots/prediksi_bulan_aplikasi_{tahun}_{bulan}.html'
            fig.write_html(grafik_prediksi_path, auto_open=False)
            grafik_prediksi_path = grafik_prediksi_path.replace('static/', '')

    return render_template(
        'aplikasi.html',
        mode_prediksi=mode_prediksi,
        tahun=tahun,
        bulan=bulan,
        hasil_prediksi=hasil_prediksi,
        grafik_prediksi_path=grafik_prediksi_path
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        depth = float(data.get('depth'))

        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model.joblib')

        features = np.array([[latitude, longitude, depth]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        result = {
            'prediction': float(prediction),
            'input': {
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True) 