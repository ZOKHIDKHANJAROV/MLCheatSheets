#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Оценка агента: несмещённые pass@k и pass^k по таблице попыток
"""

import pandas as pd
import numpy as np

from math import comb

KS = (1, 3, 5)


def pass_at_k(n, c, k):
    # хотя бы одна успешная среди k попыток: 1 − C(n−c, k)/C(n, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    # все k попыток успешны: C(c, k)/C(n, k); при c < k comb возвращает 0
    return comb(c, k) / comb(n, k)


def main():
    df = pd.read_csv('input.csv')
    p_hat = df['c'].sum() / df['n'].sum()           # общая доля успеха — для сравнения

    rows = []
    for k in KS:
        at = np.mean([pass_at_k(n, c, k) for n, c in zip(df['n'], df['c'])])
        hat = np.mean([pass_hat_k(n, c, k) for n, c in zip(df['n'], df['c'])])
        naive = p_hat ** k                          # наивная оценка надёжности «по общей доле»
        print(f"k={k}: pass@k={at:.4f}  pass^k={hat:.4f}  (наивно p̂^k = {naive:.4f})")
        rows.append({'k': k, 'pass_at_k': at, 'pass_hat_k': hat})

    result = pd.DataFrame(rows)
    assert result['pass_at_k'].is_monotonic_increasing
    assert result['pass_hat_k'].is_monotonic_decreasing
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
