#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: LIME вручную: взвешенный локальный линейный суррогат
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

SCALE = np.array([1.0, 0.5, 2.0])
KERNEL_WIDTH = 0.75 * np.sqrt(3)


def black_box(X):
    t = 2 * X[:, 0] ** 2 + X[:, 1] - X[:, 0] * X[:, 2] - 1
    return 1 / (1 + np.exp(-t))


def main():
    points = pd.read_csv('input.csv')
    E = pd.read_csv('noise.csv')[['e1', 'e2', 'e3']].to_numpy()
    rows = []
    for pt in points.itertuples():
        x0 = np.array([pt.x1, pt.x2, pt.x3])
        Z = x0 + E * SCALE                                   # 1) возмущения
        p = black_box(Z)                                     # 2) ответы чёрного ящика
        pi = np.exp(-(E ** 2).sum(axis=1) / KERNEL_WIDTH ** 2)   # 3) веса близости

        # 4) взвешенный МНК: строки умножаем на √π
        A = np.column_stack([np.ones(len(E)), E])
        sw = np.sqrt(pi)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], p * sw, rcond=None)
        p_hat = A @ coef
        p_bar = np.average(p, weights=pi)
        r2 = 1 - np.sum(pi * (p - p_hat) ** 2) / np.sum(pi * (p - p_bar) ** 2)   # 5)

        lr = LinearRegression().fit(E, p, sample_weight=pi)
        assert np.allclose(lr.coef_, coef[1:]) and np.isclose(lr.intercept_, coef[0])
        assert np.isclose(lr.score(E, p, sample_weight=pi), r2)

        # Проверка смысла: градиент f в x0, умноженный на масштаб признаков
        f0 = black_box(x0[None])[0]
        grad = f0 * (1 - f0) * np.array([4 * x0[0] - x0[2], 1.0, -x0[0]]) * SCALE
        print(f"id={pt.id}: LIME {coef[1:].round(3)}  ∇f·s {grad.round(3)}  R² {r2:.3f}")

        rows.append({'id': pt.id, 'f_x0': f0, 'intercept': coef[0],
                     'w1': coef[1], 'w2': coef[2], 'w3': coef[3], 'r2': r2})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
