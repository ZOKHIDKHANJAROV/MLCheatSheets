#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Multi-head attention: разрезание на головы, склейка и W_O
"""

import pandas as pd
import numpy as np

import json

H_HEADS = 2


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_heads(M, h):
    n, d = M.shape
    return M.reshape(n, h, d // h).transpose(1, 0, 2)      # (n, d_model) → (h, n, d_k)


def main():
    df = pd.read_csv('input.csv')
    X = df[[f'x{j}' for j in range(4)]].to_numpy()
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    n, d_model = X.shape
    d_k = d_model // H_HEADS

    Q, K, V = (split_heads(X @ w[name], H_HEADS) for name in ['W_Q', 'W_K', 'W_V'])
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))   # (h, n, n) — все головы одним умножением
    heads = A @ V                                          # (h, n, d_k)
    concat = heads.transpose(1, 0, 2).reshape(n, d_model)  # склейка голов
    O = concat @ w['W_O']

    # Проверка циклом по головам со срезами столбцов
    for i in range(H_HEADS):
        cols = slice(i * d_k, (i + 1) * d_k)
        q, k, v = (X @ w[name][:, cols] for name in ['W_Q', 'W_K', 'W_V'])
        assert np.allclose(softmax(q @ k.T / np.sqrt(d_k)) @ v, concat[:, cols])

    print(f"параметров (без смещений): 4·d² = {4 * d_model ** 2} — не зависит от числа голов")
    vocab = df['token'].tolist()
    result = pd.DataFrame({'token': vocab,
                           'top_head1': [vocab[j] for j in A[0].argmax(axis=1)],
                           'top_head2': [vocab[j] for j in A[1].argmax(axis=1)]})
    for j in range(d_model):
        result[f'o{j}'] = O[:, j]
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
