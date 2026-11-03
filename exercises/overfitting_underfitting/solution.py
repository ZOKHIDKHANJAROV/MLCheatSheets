#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Кривые обучения: недообучение и переобучение
"""

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

MODELS = {
    'linear': LinearRegression(),
    'tree_d3': DecisionTreeRegressor(max_depth=3, random_state=0),
    'tree_full': DecisionTreeRegressor(random_state=0),
}
SIZES = [25, 50, 100, 200, 400]


def main():
    df = pd.read_csv('input.csv')
    train, val = df[df['split'] == 'train'], df[df['split'] == 'val']
    X_val, y_val = val[['x']].to_numpy(), val['y'].to_numpy()

    rows = []
    for name, model in MODELS.items():
        for n in SIZES:
            part = train.iloc[:n]
            X, y = part[['x']].to_numpy(), part['y'].to_numpy()
            fitted = clone(model).fit(X, y)
            tr = mean_squared_error(y, fitted.predict(X))
            va = mean_squared_error(y_val, fitted.predict(X_val))
            rows.append({'model': name, 'n_train': n, 'train_mse': tr, 'val_mse': va, 'gap': va - tr})

    result = pd.DataFrame(rows).round(4)
    print(result.pivot(index='n_train', columns='model', values='val_mse'))
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
