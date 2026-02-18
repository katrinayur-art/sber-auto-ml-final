"""
API для предсказания целевых действий
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

app = Flask(__name__)

# Загрузка модели
MODEL_PATH = '../models/'

try:
    model = joblib.load(f'{MODEL_PATH}model.pkl')
    encoders = joblib.load(f'{MODEL_PATH}encoders.pkl')
    feature_cols = joblib.load(f'{MODEL_PATH}feature_cols.pkl')
    print("✅ Модель загружена")
except Exception as e:
    print(f"⚠️ Ошибка загрузки модели: {e}")
    model = None


def preprocess_input(data):
    """Предобработка входных данных"""
    df = pd.DataFrame([data])
    
    # Кодирование категориальных признаков
    cat_cols = ['utm_source', 'utm_medium', 'device_category', 
                'device_os', 'geo_country', 'geo_city']
    
    for col in cat_cols:
        if col in df.columns and col in encoders:
            df[col] = df[col].fillna('unknown')
            # Проверяем, есть ли значение в обученном энкодере
            df[col] = df[col].apply(
                lambda x: x if x in encoders[col].classes_ else 'unknown'
            )
            # Добавляем 'unknown' если его нет в классах
            if 'unknown' not in encoders[col].classes_:
                encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')
            df[col + '_encoded'] = encoders[col].transform(df[col])
        else:
            df[col + '_encoded'] = 0
    
    # Временные признаки (значения по умолчанию)
    df['visit_dayofweek'] = 0
    df['visit_month'] = 1
    df['is_weekend'] = 0
    df['total_hits'] = 0
    df['unique_events'] = 0
    
    # Добавляем visit_number если его нет
    if 'visit_number' not in df.columns:
        df['visit_number'] = 1  # значение по умолчанию
    
    return df[feature_cols]


@app.route('/')
def index():
    return jsonify({
        "service": "СберАвтоподписка ML API",
        "status": "running",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание целевого действия"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data"}), 400
        
        # Предобработка
        X = preprocess_input(data)
        
        # Предсказание
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "will_convert": bool(prediction == 1)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
