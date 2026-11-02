#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: валидация временного ряда
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def main():
    df = pd.read_csv('input.csv', parse_dates=['date'])

    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    # ИСПРАВЛЕНО: скользящее среднее только по прошлым 3 дням (t-3 … t-1).
    # Было: rolling(3, center=True) — окно t-1 … t+1 включало сам ответ y[t] и следующий день.
    df['rolling_mean_3'] = df['sales'].shift(1).rolling(3).mean()
    df['dayofweek'] = df['date'].dt.day_of_week
    df = df.dropna().reset_index(drop=True)

    FEATURES = ['lag_1', 'lag_7', 'rolling_mean_3', 'dayofweek']
    X, y = df[FEATURES], df['sales']

    # ИСПРАВЛЕНО: TimeSeriesSplit — обучение всегда на прошлом, проверка на следующем отрезке.
    # Было: KFold(shuffle=True) — модель училась на днях, идущих после тестовых.
    cv = TimeSeriesSplit(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        assert train_idx.max() < test_idx.min()          # прошлое → будущее
        model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
        mae = mean_absolute_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        rows.append({'fold': fold, 'mae': mae})
    result = pd.DataFrame(rows).round(4)
    print(result, f"\nСредняя MAE: {result['mae'].mean():.4f}")
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
