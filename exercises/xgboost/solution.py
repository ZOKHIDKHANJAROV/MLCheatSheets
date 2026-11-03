#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: XGBoost вручную: градиенты, гессианы, прирост и веса листьев
"""

import pandas as pd
import numpy as np

LAMBDA, GAMMA = 1.0, 0.1
FEATURES = ['x1', 'x2']


def score(G, H):
    return G ** 2 / (H + LAMBDA)


def best_split(X, g, h):
    G, H = g.sum(), h.sum()
    candidates = []
    for j, name in enumerate(FEATURES):
        values = np.unique(X[:, j])
        for thr in (values[:-1] + values[1:]) / 2:
            left = X[:, j] < thr
            GL, HL = g[left].sum(), h[left].sum()
            gain = 0.5 * (score(GL, HL) + score(G - GL, H - HL) - score(G, H)) - GAMMA
            candidates.append((gain, name, j, thr))
    best = max(candidates, key=lambda c: c[0])        # max берёт первый из равных
    gains = sorted((c[0] for c in candidates), reverse=True)
    assert gains[0] - gains[1] > 1e-6                 # победитель однозначен
    return best


def main():
    df = pd.read_csv('input.csv')
    X = df[FEATURES].to_numpy(dtype=float)
    p = 1 / (1 + np.exp(-df['F'].to_numpy()))
    g = p - df['y'].to_numpy()                        # градиент log loss по счёту
    h = p * (1 - p)                                   # гессиан

    nodes = {0: np.arange(len(df))}                   # номер узла → индексы объектов
    rows = []
    for node in range(7):
        idx = nodes[node]
        G, H = g[idx].sum(), h[idx].sum()
        row = {'node': node, 'depth': int(np.log2(node + 1)), 'n': len(idx),
               'G': G, 'H': H, 'weight': -G / (H + LAMBDA)}
        if node < 3:                                  # внутренние узлы: корень и его потомки
            gain, name, j, thr = best_split(X[idx], g[idx], h[idx])
            assert gain > 0, 'разбиение должно окупать γ'
            left = X[idx, j] < thr
            nodes[2 * node + 1], nodes[2 * node + 2] = idx[left], idx[~left]
            row.update({'feature': name, 'threshold': thr, 'gain': gain})
        else:
            row.update({'feature': 'leaf', 'threshold': np.nan, 'gain': np.nan})
        rows.append(row)

    result = pd.DataFrame(rows)
    print(result.round(4))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
