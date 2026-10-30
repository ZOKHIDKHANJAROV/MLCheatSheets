#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Значения Шепли вручную
"""

import pandas as pd
import numpy as np

from itertools import combinations
from math import factorial


def f(x):
    return 3 * x[0] + 2 * x[1] * x[2] - x[2] ** 2


def shapley(x, base):
    n = len(x)
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in combinations(others, k):
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                with_i, without_i = base.copy(), base.copy()
                for j in S:
                    with_i[j] = without_i[j] = x[j]
                with_i[i] = x[i]
                phi[i] += weight * (f(with_i) - f(without_i))
    return phi


def main():
    base = pd.read_csv('background.csv')[['x1', 'x2', 'x3']].mean().to_numpy()
    df = pd.read_csv('input.csv')
    rows = []
    for _, r in df.iterrows():
        x = r[['x1', 'x2', 'x3']].to_numpy(dtype=float)
        phi = shapley(x, base)
        assert np.isclose(phi.sum(), f(x) - f(base))      # аддитивность
        rows.append({'id': int(r['id']), 'f_x': f(x), 'f_base': f(base), 'phi1': phi[0], 'phi2': phi[1], 'phi3': phi[2]})
    print("аддитивность выполнена для всех объектов")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
