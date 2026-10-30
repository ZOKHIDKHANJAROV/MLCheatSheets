#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Scaled dot-product attention
"""

import pandas as pd
import numpy as np

def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    df = pd.read_csv('input.csv')
    Q = df[['q0', 'q1', 'q2']].to_numpy()
    K = df[['k0', 'k1', 'k2']].to_numpy()
    V = df[['v0', 'v1', 'v2']].to_numpy()
    d = Q.shape[1]

    A = softmax(Q @ K.T / np.sqrt(d))       # 4 × 4, строки суммируются в 1
    O = A @ V                               # 4 × 3
    assert np.allclose(A.sum(axis=1), 1)

    # Бонус: каузальная маска (как в декодере GPT)
    masked = np.where(np.triu(np.ones((4, 4)), k=1) == 1, -np.inf, Q @ K.T / np.sqrt(d))
    print("веса с каузальной маской:\n", softmax(masked).round(3))

    result = pd.DataFrame({'token': df['token']})
    for j in range(4):
        result[f'w{j}'] = A[:, j]
    for j in range(3):
        result[f'o{j}'] = O[:, j]
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
