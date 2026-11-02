#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 6: удаление коррелирующих признаков
"""

import pandas as pd
import numpy as np

THRESHOLD = 0.8


def main():
    df = pd.read_csv('input.csv')
    corr = df.corr().abs()
    cols = list(df.columns)

    # Вручную: двойной цикл по парам i < j
    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > THRESHOLD:
                print(f"|r({cols[i]}, {cols[j]})| = {corr.iloc[i, j]:.3f} → удаляем {cols[j]}")
                to_drop.add(cols[j])

    # То же через маску верхнего треугольника
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    assert to_drop == {c for c in upper.columns if (upper[c] > THRESHOLD).any()}

    kept = [c for c in cols if c not in to_drop]
    return pd.DataFrame({'feature': kept})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
