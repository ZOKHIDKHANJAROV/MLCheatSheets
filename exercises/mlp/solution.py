#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Прямой проход многослойного перцептрона
"""

import pandas as pd
import numpy as np

import json


def main():
    df = pd.read_csv('input.csv')
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    X = df[['x1', 'x2']].to_numpy()                     # n × 2

    H = np.tanh(X @ w['W1'].T + w['b1'])                # n × 3 — скрытый слой
    p = 1 / (1 + np.exp(-(H @ w['W2'].T + w['b2'])))    # n × 1 — выход
    p = p[:, 0]

    n_params = sum(v.size for v in w.values())
    print("параметров в сети:", n_params)
    return pd.DataFrame({'id': df['id'], 'h1': H[:, 0], 'h2': H[:, 1], 'h3': H[:, 2],
                         'p': p, 'pred': (p >= 0.5).astype(int)}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
