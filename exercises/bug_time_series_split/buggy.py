import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

df = pd.read_csv('input.csv', parse_dates=['date'])

# Признаки: лаги, скользящее среднее за 3 дня, день недели
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['rolling_mean_3'] = df['sales'].rolling(3, center=True).mean()
df['dayofweek'] = df['date'].dt.day_of_week
df = df.dropna().reset_index(drop=True)

FEATURES = ['lag_1', 'lag_7', 'rolling_mean_3', 'dayofweek']
X, y = df[FEATURES], df['sales']

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
    model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
    mae = mean_absolute_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
    rows.append({'fold': fold, 'mae': round(mae, 4)})
    print(f'Фолд {fold}: MAE = {mae:.4f}')

result = pd.DataFrame(rows)
print(f'Средняя MAE: {result["mae"].mean():.4f}')
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
