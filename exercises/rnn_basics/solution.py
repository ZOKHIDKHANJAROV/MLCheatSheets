#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Простая RNN: прямой проход и BPTT-градиенты
"""

import pandas as pd
import numpy as np

import json

ORDER = ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']


def forward(w, X, target):
    hs = [np.zeros(w['W_hh'].shape[0])]
    ys = []
    for x in X:
        hs.append(np.tanh(w['W_hh'] @ hs[-1] + w['W_xh'] @ x + w['b_h']))   # одни веса на всех шагах
        ys.append(float((w['W_hy'] @ hs[-1] + w['b_y'])[0]))
    ys = np.array(ys)
    return 0.5 * np.sum((ys - target) ** 2), hs, ys


def main():
    df = pd.read_csv('input.csv')
    X, target = df[['x1', 'x2']].to_numpy(), df['target'].to_numpy()
    w = {k: np.array(v, dtype=float) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

    loss, hs, ys = forward(w, X, target)
    g = {k: np.zeros_like(v) for k, v in w.items()}
    dh_next = np.zeros_like(hs[0])
    for t in reversed(range(len(X))):              # BPTT: от последнего шага к первому
        dy = ys[t] - target[t]
        g['W_hy'] += dy * hs[t + 1][None, :]
        g['b_y'] += dy
        dh = w['W_hy'][0] * dy + dh_next            # градиент от выхода + «из будущего»
        da = dh * (1 - hs[t + 1] ** 2)              # через tanh
        g['W_xh'] += np.outer(da, X[t])
        g['W_hh'] += np.outer(da, hs[t])
        g['b_h'] += da
        dh_next = w['W_hh'].T @ da

    # Проверка конечными разностями
    eps = 1e-6
    for k in ORDER:
        for idx in np.ndindex(w[k].shape):
            wp = {n: v.copy() for n, v in w.items()}; wp[k][idx] += eps
            wm = {n: v.copy() for n, v in w.items()}; wm[k][idx] -= eps
            num = (forward(wp, X, target)[0] - forward(wm, X, target)[0]) / (2 * eps)
            assert abs(num - g[k][idx]) < 1e-6, (k, idx)
    print(f"loss = {loss:.4f}; градиенты совпали с численными")

    rows = [{'param': 'loss', 'i': 0, 'j': 0, 'value': loss}]
    for k in ORDER:
        G = g[k].reshape(len(g[k]), -1) if g[k].ndim == 1 else g[k]
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                rows.append({'param': k, 'i': i, 'j': j, 'value': G[i, j]})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
