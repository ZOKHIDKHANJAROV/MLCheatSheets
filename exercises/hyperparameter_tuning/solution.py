#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Подбор глубины дерева кросс-валидацией
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor


def main():
    df = pd.read_csv('input.csv')
    X, y = df[['x1', 'x2', 'x3', 'x4']], df['y']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for depth in range(1, 9):
        scores = cross_val_score(DecisionTreeRegressor(max_depth=depth, random_state=0), X, y,
                                 cv=kf, scoring='neg_mean_squared_error')
        rows.append({'max_depth': depth, 'cv_mse': -scores.mean()})
    result = pd.DataFrame(rows)
    result['is_best'] = (result['cv_mse'] == result['cv_mse'].min()).astype(int)
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
