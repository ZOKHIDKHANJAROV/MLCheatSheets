#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Градиентный бустинг: прокат велосипедов
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

FEATURES = ['temperature', 'hour', 'weekend', 'humidity']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
    gb.fit(train[FEATURES], train['rides'])
    pred = gb.predict(test[FEATURES])

    # Сравнение с линейной моделью: бустинг ловит нелинейность (пики по часам)
    lin = LinearRegression().fit(train[FEATURES], train['rides'])
    rmse = lambda p: np.sqrt(mean_squared_error(test['rides'], p))
    print(f"RMSE бустинг: {rmse(pred):.1f}   RMSE линейная регрессия: {rmse(lin.predict(test[FEATURES])):.1f}")

    return pd.DataFrame({'id': test['id'], 'rides_pred': pred.round(2)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
