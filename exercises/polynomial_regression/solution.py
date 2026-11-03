#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Полиномиальная регрессия: выбор степени по кросс-валидации
"""

import pandas as pd
import numpy as np

from math import comb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def main():
    df = pd.read_csv('input.csv')
    X, y = df[['x1', 'x2']].to_numpy(), df['y'].to_numpy()
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    rows = []
    for d in range(1, 7):
        pipe = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                             StandardScaler(), LinearRegression())
        pipe.fit(X, y)
        n_features = pipe[0].n_output_features_
        assert n_features == comb(2 + d, d) - 1           # C(n + d, d) без константы
        train_mse = mean_squared_error(y, pipe.predict(X))
        cv_mse = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
        rows.append({'degree': d, 'n_features': n_features, 'train_mse': train_mse, 'cv_mse': cv_mse})

    result = pd.DataFrame(rows)
    print("Лучшая степень по CV:", int(result.loc[result['cv_mse'].idxmin(), 'degree']))
    print(make_pipeline(PolynomialFeatures(3, include_bias=False)).fit(X)[0]
          .get_feature_names_out(['x1', 'x2']))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
