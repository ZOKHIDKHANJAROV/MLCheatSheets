#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: TCN: каузальная дилатированная свёртка и рецептивное поле
"""

import pandas as pd
import numpy as np

import json

K = 3


def causal_conv(x, w, b, d):
    pad = (K - 1) * d
    xp = np.concatenate([np.zeros(pad), x])                # нули только слева — будущее недоступно
    return np.array([b + sum(w[j] * xp[pad + t - j * d] for j in range(K)) for t in range(len(x))])


def forward(x, p):
    relu = lambda v: np.maximum(v, 0)
    h1 = relu(causal_conv(x, p['w1'], p['b1'], 1))
    h2 = relu(causal_conv(h1, p['w2'], p['b2'], 2))
    out = x + causal_conv(h2, p['w3'], p['b3'], 4)        # остаточная связь
    return h1, h2, out


def main():
    df = pd.read_csv('input.csv')
    x = df['x'].to_numpy(dtype=float)
    p = json.load(open('weights.json', encoding='utf-8'))
    h1, h2, out = forward(x, p)

    R = 1 + (K - 1) * (1 + 2 + 4)
    n_inputs = np.minimum(np.arange(len(x)) + 1, R)        # начало ряда обрезает окно
    print("рецептивное поле R =", R)

    # Проверка каузальности: меняем последнюю точку — ранние выходы не меняются
    x2 = x.copy(); x2[-1] += 10
    assert np.allclose(forward(x2, p)[2][:-1], out[:-1])

    return pd.DataFrame({'t': df['t'], 'h1': h1, 'h2': h2, 'out': out, 'n_inputs': n_inputs}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
