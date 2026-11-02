#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 1: удаление низкодисперсных признаков (Variance Threshold)
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold

THRESHOLD = 0.5


def main():
    df = pd.read_csv('input.csv')

    # Дисперсия с делением на n (ddof=0)
    variances = ((df - df.mean()) ** 2).mean()
    assert np.allclose(variances, df.var(ddof=0))
    print("Дисперсии:", variances.round(4).to_dict())

    kept = variances.index[variances >= THRESHOLD]

    # Сверка: VarianceThreshold оставляет признаки с дисперсией > threshold
    # (здесь ни одна дисперсия не равна 0.5, так что ≥ и > совпадают)
    selector = VarianceThreshold(threshold=THRESHOLD).fit(df)
    assert list(df.columns[selector.get_support()]) == list(kept)

    return pd.DataFrame({'feature': kept})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
