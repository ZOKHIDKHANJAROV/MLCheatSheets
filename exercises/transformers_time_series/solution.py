#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Патчи PatchTST и каузальное внимание по ним
"""

import pandas as pd
import numpy as np

import json

P, S = 8, 4


def main():
    y = pd.read_csv('input.csv')['y'].to_numpy(dtype=float)
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

    z = (y - y.mean()) / y.std()                                   # RevIN: нормализация окна (ddof=0)
    starts = np.arange(0, len(z) - P + 1, S)
    patches = np.stack([z[s:s + P] for s in starts])               # (11, 8): токен = кусок ряда
    N, d = len(patches), w['W_q'].shape[0]

    E = patches @ w['W_emb'] + w['b_emb']
    Q, K, V = E @ w['W_q'], E @ w['W_k'], E @ w['W_v']
    scores = Q @ K.T / np.sqrt(d)
    scores[np.triu(np.ones((N, N), dtype=bool), k=1)] = -np.inf   # будущее закрыто
    A = np.exp(scores - scores.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    ctx = A @ V

    assert np.allclose(np.triu(A, k=1), 0) and np.allclose(A.sum(axis=1), 1)
    print("токенов:", N, "вместо", len(z), "точек; пар для внимания:", N * N, "вместо", len(z) ** 2)

    return pd.DataFrame({'patch': np.arange(N), 't_start': starts, 't_end': starts + P - 1,
                         'top_key': A.argmax(axis=1), 'top_weight': A.max(axis=1),
                         'self_weight': np.diag(A), 'ctx1': ctx[:, 0], 'ctx2': ctx[:, 1]}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
