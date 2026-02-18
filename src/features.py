"""
Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_features(sessions_df, hits_df):
    """Создание признаков для модели"""
    df = sessions_df.copy()
    
    # Временные признаки
    df['visit_dayofweek'] = df['visit_date'].dt.dayofweek
    df['visit_month'] = df['visit_date'].dt.month
    df['is_weekend'] = (df['visit_dayofweek'] >= 5).astype(int)
    
    # Признаки из hits
    hits_stats = hits_df.groupby('session_id').agg({
        'hit_number': 'count',
        'event_action': 'nunique'
    }).rename(columns={
        'hit_number': 'total_hits',
        'event_action': 'unique_events'
    })
    
    df = df.merge(hits_stats, left_on='session_id', right_index=True, how='left')
    df['total_hits'] = df['total_hits'].fillna(0)
    df['unique_events'] = df['unique_events'].fillna(0)
    
    return df


def encode_categorical(df, cat_cols):
    """Кодирование категориальных признаков"""
    df = df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Заполняем NaN
            df[col] = df[col].fillna('unknown')
            # Кодируем только топ-N категорий
            top_categories = df[col].value_counts().head(50).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
            df[col + '_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders
