"""
Обучение модели
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


def prepare_data(sessions_df, hits_df):
    """Подготовка данных для обучения"""
    from data_loader import create_target_variable
    from features import create_features, encode_categorical
    
    # Создаем целевую переменную
    df = create_target_variable(sessions_df, hits_df)
    
    # Создаем признаки
    df = create_features(df, hits_df)
    
    # Категориальные колонки
    cat_cols = ['utm_source', 'utm_medium', 'device_category', 
                'device_os', 'geo_country', 'geo_city']
    
    df, encoders = encode_categorical(df, cat_cols)
    
    # Выбираем признаки для модели
    feature_cols = [col for col in df.columns if col.endswith('_encoded')] + \
                   ['visit_number', 'visit_dayofweek', 'visit_month', 
                    'is_weekend', 'total_hits', 'unique_events']
    
    # Удаляем строки с пропусками в целевой переменной
    df = df.dropna(subset=['target'])
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, feature_cols, encoders


def train_model(X, y):
    """Обучение модели"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
    
    # Обучаем RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\\n🎯 ROC-AUC: {roc_auc:.4f}")
    print(f"\\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    return model, roc_auc


def save_model(model, encoders, feature_cols, path='../models/'):
    """Сохранение модели"""
    import os
    os.makedirs(path, exist_ok=True)
    
    joblib.dump(model, f'{path}model.pkl')
    joblib.dump(encoders, f'{path}encoders.pkl')
    joblib.dump(feature_cols, f'{path}feature_cols.pkl')
    
    print(f"✅ Модель сохранена в {path}")


if __name__ == '__main__':
    from data_loader import load_data, clean_data
    
    print("📊 Загрузка данных...")
    sessions = load_data('ga_sessions')
    hits = load_data('ga_hits')
    
    print("🧹 Очистка данных...")
    sessions = clean_data(sessions)
    hits = clean_data(hits)
    
    print("🔧 Подготовка признаков...")
    X, y, feature_cols, encoders = prepare_data(sessions, hits)
    
    print("🚀 Обучение модели...")
    model, roc_auc = train_model(X, y)
    
    print("💾 Сохранение модели...")
    save_model(model, encoders, feature_cols)
    
    print("\\n✅ Готово!")
