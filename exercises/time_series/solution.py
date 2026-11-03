#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Лаговые признаки без утечки и walk-forward проверка
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

FEATURES = ['lag_1', 'lag_7', 'roll_mean_7']
TEST_DAYS, FOLD = 28, 7


def make_features(df):
    out = df.copy()
    out['lag_1'] = out['sales'].shift(1)
    out['lag_7'] = out['sales'].shift(7)
    out['roll_mean_7'] = out['sales'].shift(1).rolling(7).mean()   # окно t−7…t−1, без sales(t)
    return out


def main():
    df = pd.read_csv('input.csv')
    data = make_features(df)

    # Проверка на утечку: признаки дня t не должны зависеть от будущего
    cut = 100
    assert np.allclose(make_features(df.iloc[:cut])[FEATURES].iloc[-1], data[FEATURES].iloc[cut - 1])

    first_test = len(data) - TEST_DAYS
    rows = []
    for k in range(TEST_DAYS // FOLD):
        start = first_test + k * FOLD
        train = data.iloc[:start].dropna(subset=FEATURES)       # только прошлое
        test = data.iloc[start:start + FOLD]
        model = LinearRegression().fit(train[FEATURES], train['sales'])
        pred = model.predict(test[FEATURES])
        rows.append({'fold': k + 1, 'test_start': test['date'].iloc[0], 'train_size': len(train),
                     'mae_model': np.mean(np.abs(test['sales'] - pred)),
                     'mae_snaive': np.mean(np.abs(test['sales'] - test['lag_7']))})

    result = pd.DataFrame(rows)
    ratio = result['mae_model'].mean() / result['mae_snaive'].mean()
    print(f"Средний MAE: модель {result['mae_model'].mean():.2f}, сезонный наивный {result['mae_snaive'].mean():.2f}"
          f"  (отношение {ratio:.2f} — меньше 1, значит модель полезна)")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
