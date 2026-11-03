#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Ячейка GRU: прямой проход в раскладке PyTorch
"""

import pandas as pd
import numpy as np

import json


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    df = pd.read_csv('input.csv')
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    h = w['h0'].copy()

    rows = []
    for t, x in zip(df['t'], df[['x1', 'x2']].to_numpy()):
        gi = w['W_ih'] @ x + w['b_ih']               # вклад входа во все 3 блока
        gh = w['W_hh'] @ h + w['b_hh']               # вклад состояния (со смещением b_hh)
        i_r, i_z, i_n = np.split(gi, 3)              # порядок PyTorch: r, z, n
        h_r, h_z, h_n = np.split(gh, 3)
        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)                   # reset gate «гасит» вклад прошлого состояния
        h = (1 - z) * n + z * h                      # z ≈ 1 → состояние почти не меняется
        rows.append({'t': t, **{f'h{k + 1}': h[k] for k in range(3)}, **{f'z{k + 1}': z[k] for k in range(3)}})
        print(f"t={t}: r = {r.round(3)}, z = {z.round(3)}")

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
