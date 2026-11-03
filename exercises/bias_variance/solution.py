#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Разложение ошибки на смещение и дисперсию
"""

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

DEPTHS = [1, 2, 3, 4, 6, 8, 12]
SIGMA2 = 0.3 ** 2          # неустранимый шум


def main():
    df = pd.read_csv('input.csv')
    grid = pd.read_csv('grid.csv')
    x_grid = grid[['x']].to_numpy()
    f_grid = grid['f'].to_numpy()
    samples = [g for _, g in df.groupby('sample')]

    rows = []
    for depth in DEPTHS:
        # 40 моделей, обученных на разных выборках, — 40 прогнозов в каждой точке сетки
        preds = np.array([
            DecisionTreeRegressor(max_depth=depth, random_state=0)
            .fit(s[['x']].to_numpy(), s['y'].to_numpy()).predict(x_grid)
            for s in samples
        ])
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - f_grid) ** 2)
        variance = preds.var(axis=0).mean()

        # Разложение точное: E[(f̂ − f)²] = bias² + variance
        assert np.isclose(np.mean((preds - f_grid) ** 2), bias2 + variance)
        rows.append({'max_depth': depth, 'bias2': bias2, 'variance': variance,
                     'expected_mse': bias2 + variance + SIGMA2})

    result = pd.DataFrame(rows).round(4)
    best = result.loc[result['expected_mse'].idxmin(), 'max_depth']
    print(f"Минимум ожидаемой ошибки — при глубине {best}")
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
