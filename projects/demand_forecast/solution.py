#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эталонное решение мини-проекта: Прогноз спроса на 4 недели
Запуск из папки проекта: python solution.py → solution_predictions.csv
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor


def features(df, t0):
    d = pd.to_datetime(df['date'])
    return pd.DataFrame({
        't': (d - t0).dt.days, 'dow': d.dt.dayofweek,
        'doy_sin': np.sin(2 * np.pi * d.dt.dayofyear / 365.25), 'doy_cos': np.cos(2 * np.pi * d.dt.dayofyear / 365.25),
        'promo': df['promo'], 'holiday': df['holiday'],
    })


def main():
    train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
    t0 = pd.to_datetime(train['date']).min()
    X, Xt = features(train, t0), features(test, t0)

    # Деревья плохо экстраполируют тренд: прогнозируем отношение к линейному тренду
    coef = np.polyfit(X['t'], train['sales'], 1)
    trend, trend_t = np.polyval(coef, X['t']), np.polyval(coef, Xt['t'])
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[1], random_state=0)

    # Валидация по времени: последние 28 дней train
    cut = len(train) - 28
    gb.fit(X.iloc[:cut], (train['sales'] / trend).iloc[:cut])
    val_pred = gb.predict(X.iloc[cut:]) * trend[cut:]
    naive = train['sales'].to_numpy()[cut - 7:cut]
    naive_pred = np.array([naive[i % 7] for i in range(28)])
    y_val = train['sales'].iloc[cut:].to_numpy()
    print(f"MAE на валидации: бустинг {np.abs(val_pred - y_val).mean():.1f}, сезонный наивный {np.abs(naive_pred - y_val).mean():.1f}")

    gb.fit(X, train['sales'] / trend)
    return pd.DataFrame({'id': test['id'], 'sales': (gb.predict(Xt) * trend_t).round(1)})


if __name__ == "__main__":
    result = main()
    result.to_csv('solution_predictions.csv', index=False)
    print("✅ Прогнозы сохранены в solution_predictions.csv")
