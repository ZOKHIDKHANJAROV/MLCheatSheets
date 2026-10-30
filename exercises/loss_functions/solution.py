#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Softmax и кросс-энтропия (численно устойчиво)
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('input.csv')
    z = df[['z0', 'z1', 'z2']].to_numpy()
    y = df['y'].to_numpy()

    # Устойчивый softmax: сдвиг на максимум не меняет результат, но убирает переполнение
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    p = exp / exp.sum(axis=1, keepdims=True)

    # Устойчивая кросс-энтропия через log-sum-exp
    logsumexp = z.max(axis=1) + np.log(exp.sum(axis=1))
    loss = logsumexp - z[np.arange(len(y)), y]

    with np.errstate(over='ignore', invalid='ignore'):
        naive = np.exp(z[3]) / np.exp(z[3]).sum()
    print("наивный softmax для строки 4:", naive, " устойчивый:", p[3].round(4))

    return pd.DataFrame({'id': df['id'], 'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2],
                         'loss': loss, 'pred': p.argmax(axis=1)}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
