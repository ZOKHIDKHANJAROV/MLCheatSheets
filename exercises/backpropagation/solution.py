#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Обратное распространение для одного нейрона
"""

import pandas as pd
import numpy as np

W1, W2, B = 0.5, -0.3, 0.1


def forward(x1, x2, y, w1=W1, w2=W2, b=B):
    z = w1 * x1 + w2 * x2 + b
    a = 1 / (1 + np.exp(-z))
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    return z, a, loss


def main():
    df = pd.read_csv('input.csv')
    x1, x2, y = df['x1'].to_numpy(), df['x2'].to_numpy(), df['y'].to_numpy()

    z, a, loss = forward(x1, x2, y)
    dz = a - y                      # ∂L/∂a · ∂a/∂z = (a − y) / (a(1 − a)) · a(1 − a)
    dw1, dw2, db = dz * x1, dz * x2, dz

    # Численная проверка градиента по w1
    eps = 1e-5
    numeric = (forward(x1, x2, y, w1=W1 + eps)[2] - forward(x1, x2, y, w1=W1 - eps)[2]) / (2 * eps)
    print("max |аналитический − численный| для dw1:", np.abs(numeric - dw1).max())

    return pd.DataFrame({'id': df['id'], 'z': z, 'a': a, 'loss': loss,
                         'dz': dz, 'dw1': dw1, 'dw2': dw2, 'db': db}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
