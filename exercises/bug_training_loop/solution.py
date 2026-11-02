#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: цикл градиентного спуска
"""

import pandas as pd
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    df = pd.read_csv('input.csv')
    X = np.column_stack([np.ones(len(df)), df[['x1', 'x2']].to_numpy()])
    y = df['y'].to_numpy()
    n = len(y)

    lr = 0.5
    epochs = 30
    w = np.zeros(3)

    rows = []
    for epoch in range(1, epochs + 1):
        p = sigmoid(X @ w)
        # ИСПРАВЛЕНО (1): градиент считается заново на каждой эпохе (присваивание, а не +=).
        # Было: grad = np.zeros(3) до цикла и grad += …, т.е. сумма градиентов всех прошлых эпох.
        # ИСПРАВЛЕНО (2): деление на n — градиент СРЕДНЕГО loss, как в формуле.
        # Было: Xᵀ(p − y) без /n — шаг в 150 раз больше задуманного.
        grad = X.T @ (p - y) / n
        w -= lr * grad
        loss = log_loss(y, sigmoid(X @ w))
        rows.append({'epoch': epoch, 'loss': loss, 'w0': w[0], 'w1': w[1], 'w2': w[2]})

    result = pd.DataFrame(rows).round(4)
    assert result['loss'].is_monotonic_decreasing      # с разумным шагом loss убывает
    print(result.iloc[[0, 9, 19, 29]])
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
