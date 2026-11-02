#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Ячейка LSTM: прямой проход по последовательности
"""

import pandas as pd
import numpy as np

import json


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    df = pd.read_csv('input.csv')
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    H = w['W_hh'].shape[1]

    h, c = np.zeros(H), np.zeros(H)
    rows = []
    for t, x in zip(df['t'], df[['x1', 'x2']].to_numpy()):
        a = w['W_ih'] @ x + w['b_ih'] + w['W_hh'] @ h + w['b_hh']   # все вентили сразу
        a_i, a_f, a_g, a_o = np.split(a, 4)                          # порядок PyTorch: i, f, g, o
        i, f, g, o = sigmoid(a_i), sigmoid(a_f), np.tanh(a_g), sigmoid(a_o)
        c = f * c + i * g            # память: часть старой + часть нового
        h = o * np.tanh(c)           # выход: отфильтрованная память
        rows.append({'t': t, **{f'h{k + 1}': h[k] for k in range(H)},
                     **{f'c{k + 1}': c[k] for k in range(H)}})
        print(f"t={t}: f = {f.round(3)} (что сохранить), i = {i.round(3)} (что записать)")

    # h всегда в (−1, 1): o ∈ (0, 1), tanh(c) ∈ (−1, 1); c ничем не ограничена
    result = pd.DataFrame(rows)
    assert (result.filter(like='h').abs() < 1).all().all()
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
