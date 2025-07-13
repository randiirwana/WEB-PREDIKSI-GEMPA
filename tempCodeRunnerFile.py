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
    return render_template('analisis.html')

@app.route('/aplikasi')
def aplikasi():
    return render_template('aplikasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form atau JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        try:
            latitude = float(data.get('latitude'))
            longitude = float(data.get('longitude'))
            depth = float(data.get('depth'))
        except Exception as e:
            logger.error(f"Input tidak valid: {str(e)}")
            return jsonify({'error': 'Input latitude, longitude, dan depth harus berupa angka'}), 400

        # Load scaler dan model
        try:
            scaler = joblib.load('scaler.joblib')
            model = joblib.load('model.joblib')
        except Exception as e:
            logger.error(f"Gagal memuat model atau scaler: {str(e)}")
            return jsonify({'error': 'Model atau scaler tidak ditemukan'}), 500

        # Preprocessing fitur
        features = np.array([[latitude, longitude, depth]])
        features_scaled = scaler.transform(features)

        # Prediksi
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        pred_prob = float(probability[model.classes_.tolist().index(prediction)])

        # Output
        result = {
            'prediction': prediction,
            'probability': pred_prob,
            'input': {
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth
            }
        }
        logger.info(f"Prediksi berhasil: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error dalam proses prediksi: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan dalam memproses request'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 