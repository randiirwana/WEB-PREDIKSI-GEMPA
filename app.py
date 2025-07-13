from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import logging
from datetime import datetime

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisis')
def analisis():
    # Baca nilai MAE dan RMSE dari file
    try:
        with open('metrics.txt') as f:
            lines = f.readlines()
            mae = float(lines[0].strip())
            rmse = float(lines[1].strip())
    except:
        mae = None
        rmse = None
    return render_template('analisis.html', mae=mae, rmse=rmse)

@app.route('/aplikasi')
def aplikasi():
    return render_template('aplikasi.html')

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
    app.run(host='0.0.0.0', port=port) 