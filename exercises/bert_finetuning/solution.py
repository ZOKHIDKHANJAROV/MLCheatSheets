#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Голова классификатора на CLS-векторах: warmup + linear decay
"""

import pandas as pd
import numpy as np

T, WARMUP, PEAK_LR = 40, 4, 0.5


def lr_at(t):
    # get_linear_schedule_with_warmup: разогрев от 0 до пика, затем линейно к нулю
    if t <= WARMUP:
        return PEAK_LR * t / WARMUP
    return PEAK_LR * (T - t) / (T - WARMUP)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    df = pd.read_csv('input.csv')
    X = df[[f'c{j}' for j in range(8)]].to_numpy()
    y = df['label'].to_numpy()
    n, k = len(y), 3
    Y = np.eye(k)[y]                                   # one-hot метки

    W, b = np.zeros((X.shape[1], k)), np.zeros(k)       # голова классификатора
    rows = []
    for t in range(1, T + 1):
        P = softmax(X @ W + b)
        loss = -np.mean(np.log(P[np.arange(n), y]))
        G = (P - Y) / n                                 # ∂L/∂z
        lr = lr_at(t)
        W -= lr * (X.T @ G)
        b -= lr * G.sum(axis=0)
        rows.append({'step': t, 'lr': lr, 'loss': loss})

    assert np.isclose(rows[0]['loss'], np.log(3))
    acc = (softmax(X @ W + b).argmax(axis=1) == y).mean()
    print(f"loss: {rows[0]['loss']:.4f} → {rows[-1]['loss']:.4f}, accuracy на обучении = {acc:.3f}")
    result = pd.DataFrame(rows)
    result['lr'] = result['lr'].round(6)
    result['loss'] = result['loss'].round(4)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
