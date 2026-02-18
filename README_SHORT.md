# СберАвтоподписка ML Project

**Stack:** Python, Pandas, Scikit-learn, XGBoost, Flask, Docker

**Задача:** Предсказание целевого действия пользователя (ROC-AUC ~0.65)

## Быстрый старт

### Docker (рекомендуется)

```bash
docker-compose up --build
```

- Jupyter: http://localhost:8888
- API: http://localhost:5000

### PyCharm локально

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Структура

- `notebooks/` - Jupyter notebooks с EDA
- `src/` - Исходный код
- `api/` - API сервис
- `data/` - Данные (ga_sessions.pkl, ga_hits.pkl)
