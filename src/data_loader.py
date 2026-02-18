"""
Загрузка и предобработка данных
"""
import pandas as pd
import numpy as np
import os


def load_data(filename, path='../data/'):
    """Загружает данные из pickle или csv"""
    filepath_pkl = os.path.join(path, filename + '.pkl')
    filepath_csv = os.path.join(path, filename + '.csv')
    
    if os.path.exists(filepath_pkl):
        return pd.read_pickle(filepath_pkl)
    elif os.path.exists(filepath_csv):
        return pd.read_csv(filepath_csv)
    else:
        raise FileNotFoundError(f"Файл {filename} не найден")


def clean_data(df):
    """Базовая очистка данных"""
    df = df.drop_duplicates()
    
    # Преобразование дат
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
    
    return df


def create_target_variable(sessions_df, hits_df):
    """Создание целевой переменной"""
    target_actions = [
        'sub_car_claim_click', 'sub_car_claim_submit_click',
        'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]
    
    target_sessions = hits_df[
        hits_df['event_action'].isin(target_actions)
    ]['session_id'].unique()
    
    sessions_df['target'] = sessions_df['session_id'].isin(target_sessions).astype(int)
    
    return sessions_df
