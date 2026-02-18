import joblib
import pandas as pd

# Загружаем feature_cols
feature_cols = joblib.load('models/feature_cols.pkl')
print("Признаки, которые ожидает модель:")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2}. {col}")

print(f"\nВсего признаков: {len(feature_cols)}")

# Группируем по типу
encoded = [col for col in feature_cols if '_encoded' in col]
other = [col for col in feature_cols if '_encoded' not in col]

print(f"\nЗакодированные категории ({len(encoded)}):")
for col in encoded:
    print(f"  - {col}")

print(f"\nДругие признаки ({len(other)}):")
for col in other:
    print(f"  - {col}")