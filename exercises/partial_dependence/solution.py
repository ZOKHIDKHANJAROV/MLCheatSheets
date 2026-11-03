#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Частичные зависимости вручную и взаимодействие признаков
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

GRID = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
FEATURES = ['x0', 'x1', 'x2']


def main():
    df = pd.read_csv('input.csv')
    X, y = df[FEATURES].to_numpy(), df['y'].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, y)
    neg = X[:, 2] < 0

    rows = []
    for j, name in [(0, 'x0'), (1, 'x1')]:
        for v in GRID:
            X_mod = X.copy()
            X_mod[:, j] = v                         # «что если» для всех объектов сразу
            ice = model.predict(X_mod)              # значения ICE-кривых в точке v
            rows.append({'feature': name, 'value': v, 'pd': ice.mean(),
                         'pd_x2_neg': ice[neg].mean(), 'pd_x2_pos': ice[~neg].mean()})

        # Сверка с библиотекой
        ref = partial_dependence(model, X, [j], custom_values={j: GRID}, method='brute')
        assert np.allclose(ref['average'][0], [r['pd'] for r in rows[-5:]])

    result = pd.DataFrame(rows)
    for name in ['x0', 'x1']:
        part = result[result['feature'] == name]
        slope = lambda c: (part[c].iloc[-1] - part[c].iloc[0]) / 3
        print(f"{name}: наклон от −1.5 до 1.5 — x2<0: {slope('pd_x2_neg'):.2f}, x2≥0: {slope('pd_x2_pos'):.2f}")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
