#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Градиентный спуск для линейной регрессии вручную
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

LR, EPOCHS = 0.1, 100


def main():
    df = pd.read_csv('input.csv')
    X = df[['area', 'age']].to_numpy(dtype=float)
    y = df['price'].to_numpy()
    n = len(y)

    # Стандартизация (ddof=0, как StandardScaler): без неё нужен крошечный lr
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    w, b = np.zeros(2), 0.0
    rows = []
    for epoch in range(1, EPOCHS + 1):
        err = X @ w + b - y                 # прямой проход: ŷ − y
        loss = np.mean(err ** 2)            # MSE до шага
        grad_w = 2 / n * X.T @ err          # ∂L/∂w
        grad_b = 2 * np.mean(err)           # ∂L/∂b
        w, b = w - LR * grad_w, b - LR * grad_b
        if epoch == 1 or epoch % 10 == 0:
            rows.append({'epoch': epoch, 'loss': loss, 'w1': w[0], 'w2': w[1], 'b': b})

    # Сверка с аналитическим решением (нормальное уравнение / LinearRegression)
    ols = LinearRegression().fit(X, y)
    print(f"GD:  w = {w.round(4)}, b = {b:.4f}")
    print(f"OLS: w = {ols.coef_.round(4)}, b = {ols.intercept_:.4f}")
    assert np.allclose(w, ols.coef_, atol=1e-3) and np.isclose(b, ols.intercept_, atol=1e-3)

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
