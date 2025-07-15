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
    df = pd.read_csv('data.csv')
    if tahun:
        df = df[df['tahun'] == int(tahun)]
    if bulan:
        df = df[df['bulan'] == int(bulan)]
    jumlah_gempa = len(df)
    mae_tahun, rmse_tahun, mae_bulan, rmse_bulan = read_metrics_jumlah_gempa()
    return render_template(
        'analisis.html',
        jumlah_gempa=jumlah_gempa,
        tahun=tahun,
        bulan=bulan,
        mae_tahun=mae_tahun,
        rmse_tahun=rmse_tahun,
        mae_bulan=mae_bulan,
        rmse_bulan=rmse_bulan
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
    jumlah_gempa_tahunan = None
    tahun_tahunan = None
    jumlah_gempa_bulanan = None
    tahun_bulanan = None
    bulan_bulanan = None

    if request.method == 'POST':
        import pandas as pd
        df = pd.read_csv('data.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['tahun'] = df['date'].dt.year
        df['bulan'] = df['date'].dt.month

        if request.form.get('cek') == 'tahunan':
            tahun_tahunan = request.form.get('tahun_tahunan')
            if tahun_tahunan:
                jumlah_gempa_tahunan = len(df[df['tahun'] == int(tahun_tahunan)])
        elif request.form.get('cek') == 'bulanan':
            tahun_bulanan = request.form.get('tahun_bulanan')
            bulan_bulanan = request.form.get('bulan_bulanan')
            if tahun_bulanan and bulan_bulanan:
                jumlah_gempa_bulanan = len(
                    df[(df['tahun'] == int(tahun_bulanan)) & (df['bulan'] == int(bulan_bulanan))]
                )

    mae_tahun, rmse_tahun, mae_bulan, rmse_bulan = read_metrics_jumlah_gempa()

    return render_template(
        'aplikasi.html',
        jumlah_gempa_tahunan=jumlah_gempa_tahunan,
        tahun_tahunan=tahun_tahunan,
        jumlah_gempa_bulanan=jumlah_gempa_bulanan,
        tahun_bulanan=tahun_bulanan,
        bulan_bulanan=bulan_bulanan,
        mae_tahun=mae_tahun,
        rmse_tahun=rmse_tahun,
        mae_bulan=mae_bulan,
        rmse_bulan=rmse_bulan
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