#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Цикл обучения как в PyTorch — на NumPy
"""

import pandas as pd
import numpy as np

import json

LR, MOMENTUM, EPOCHS = 0.1, 0.9, 30


def forward(p, X):
    h = np.tanh(X @ p['W1'].T + p['b1'])             # nn.Linear(2, 3) + nn.Tanh()
    return h, h @ p['W2'].T + p['b2']                # nn.Linear(3, 1)


def loss_fn(p, X, y):
    return np.mean((forward(p, X)[1] - y) ** 2)      # nn.MSELoss()


def backward(p, X, y):
    # loss.backward(): цепное правило от L к каждому параметру
    h, y_hat = forward(p, X)
    d_yhat = 2 * (y_hat - y) / len(y)                # ∂L/∂ŷ  (m × 1)
    d_h = d_yhat @ p['W2']                           # ∂L/∂h  (m × 3)
    d_z = d_h * (1 - h ** 2)                         # через tanh
    return {'W2': d_yhat.T @ h, 'b2': d_yhat.sum(axis=0),
            'W1': d_z.T @ X, 'b1': d_z.sum(axis=0)}


def main():
    df = pd.read_csv('input.csv')
    params = {k: np.array(v, dtype=float) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    batches = [(g[['x1', 'x2']].to_numpy(), g[['y']].to_numpy()) for _, g in df.groupby('batch', sort=True)]

    # Численная проверка градиента на первом батче (до обучения)
    X0, y0 = batches[0]
    grads = backward(params, X0, y0)
    eps = 1e-6
    for (i, j) in [(0, 0), (2, 1)]:
        plus = {k: v.copy() for k, v in params.items()}; plus['W1'][i, j] += eps
        minus = {k: v.copy() for k, v in params.items()}; minus['W1'][i, j] -= eps
        numeric = (loss_fn(plus, X0, y0) - loss_fn(minus, X0, y0)) / (2 * eps)
        assert np.isclose(numeric, grads['W1'][i, j], atol=1e-6)

    velocity = {k: np.zeros_like(v) for k, v in params.items()}  # буферы момента
    rows = []
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for X, y in batches:
            running += loss_fn(params, X, y)          # forward + loss (до шага)
            grads = backward(params, X, y)            # zero_grad + backward: градиенты «с нуля»
            for k in params:                          # optimizer.step()
                velocity[k] = MOMENTUM * velocity[k] + grads[k]
                params[k] -= LR * velocity[k]
        rows.append({'epoch': epoch, 'loss': running / len(batches)})

    print(f"loss: эпоха 1 = {rows[0]['loss']:.4f} → эпоха {EPOCHS} = {rows[-1]['loss']:.4f}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
