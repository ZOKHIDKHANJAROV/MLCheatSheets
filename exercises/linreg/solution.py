#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Линейная регрессия: цена квартиры
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

FEATURES = ['area', 'rooms', 'age', 'distance']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    model = LinearRegression().fit(train[FEATURES], train['price'])
    print("Коэффициенты:")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name:>9}: {coef:+.3f}")
    print(f"  intercept: {model.intercept_:+.3f}")

    pred = model.predict(test[FEATURES])
    print(f"RMSE на тесте: {np.sqrt(mean_squared_error(test['price'], pred)):.2f}")

    return pd.DataFrame({'id': test['id'], 'price_pred': pred.round(2)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
