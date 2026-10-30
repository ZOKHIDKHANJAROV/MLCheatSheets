#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Дрейф данных: Population Stability Index
"""

import pandas as pd
import numpy as np

FEATURES = ['age', 'income', 'sessions']


def psi(ref, cur, eps=1e-4):
    edges = np.quantile(ref, np.arange(0.1, 1.0, 0.1))
    p = np.bincount(np.searchsorted(edges, ref, side='right'), minlength=10) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side='right'), minlength=10) / len(cur)
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    df = pd.read_csv('input.csv')
    ref, cur = df[df['sample'] == 'reference'], df[df['sample'] == 'current']
    rows = []
    for f in FEATURES:
        v = psi(ref[f].to_numpy(), cur[f].to_numpy())
        status = 'stable' if v < 0.1 else 'moderate' if v < 0.25 else 'drift'
        rows.append({'feature': f, 'psi': round(v, 4), 'status': status})
        print(f"{f:>9}: PSI = {v:.4f} → {status}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
