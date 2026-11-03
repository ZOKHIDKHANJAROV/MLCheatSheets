#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Позиционное кодирование и блок трансформера: Post-LN и Pre-LN
"""

import pandas as pd
import numpy as np

import json

EPS = 1e-5


def positional_encoding(n, d):
    pos, i = np.arange(n)[:, None], np.arange(0, d, 2)[None, :]
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos / 10000 ** (i / d))
    pe[:, 1::2] = np.cos(pos / 10000 ** (i / d))
    return pe


def layer_norm(z, gamma, beta):
    mu = z.mean(axis=-1, keepdims=True)
    var = z.var(axis=-1, keepdims=True)                   # смещённая дисперсия, как в nn.LayerNorm
    return gamma * (z - mu) / np.sqrt(var + EPS) + beta


def main():
    df = pd.read_csv('input.csv')
    E = df[[f'e{j}' for j in range(6)]].to_numpy()
    p = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    n, d = E.shape

    ffn = lambda z: np.maximum(0, z @ p['W1'] + p['b1']) @ p['W2'] + p['b2']
    ln = lambda z: layer_norm(z, p['gamma'], p['beta'])

    pe = positional_encoding(n, d)
    x = E + pe
    post = ln(x + ffn(x))          # Post-LN: нормализация после сложения
    pre = x + ffn(ln(x))           # Pre-LN: «дорога» остаточной связи остаётся чистой

    # Post-LN нормирует выход: при γ = 1, β = 0 у каждой строки среднее 0 и дисперсия 1
    assert np.allclose(layer_norm(x, 1, 0).mean(axis=1), 0)
    print("норма строк post:", np.linalg.norm(post, axis=1).round(3), " pre:", np.linalg.norm(pre, axis=1).round(3))

    rows = [{'token': df['token'][t], 'pos': t, 'dim': j, 'pe': pe[t, j], 'post': post[t, j], 'pre': pre[t, j]}
            for t in range(n) for j in range(d)]
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
