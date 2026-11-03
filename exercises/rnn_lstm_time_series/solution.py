#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Окна, скейлер по train и прогноз рекуррентной сетью
"""

import pandas as pd
import numpy as np

import json

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

WINDOW, N_TRAIN = 12, 48


def last_hidden_state(window, w):
    h = np.zeros(w['W_hh'].shape[0])
    for x in window:                                     # читаем окно слева направо
        h = np.tanh(w['W_ih'][:, 0] * x + w['b_ih'] + w['W_hh'] @ h + w['b_hh'])
    return h


def main():
    df = pd.read_csv('input.csv')
    y = df['y'].to_numpy(dtype=float)
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

    scaler = MinMaxScaler().fit(y[:N_TRAIN].reshape(-1, 1))   # только train!
    z = scaler.transform(y.reshape(-1, 1)).ravel()

    targets = np.arange(WINDOW, len(y))
    X = np.array([z[i - WINDOW:i] for i in targets])           # (60, 12) → для PyTorch (60, 12, 1)
    print("окна:", X.shape, "| масштабированный тест: max =", z[N_TRAIN:].max().round(3))
    H = np.array([last_hidden_state(x, w) for x in X])         # (60, 4)

    is_train = targets < N_TRAIN
    head = LinearRegression().fit(H[is_train], z[targets[is_train]])
    pred = scaler.inverse_transform(head.predict(H[~is_train]).reshape(-1, 1)).ravel()

    t_test = targets[~is_train]
    result = pd.DataFrame({'t': df['t'].to_numpy()[t_test], 'y_true': y[t_test],
                           'y_naive': y[t_test - 1], 'y_rnn': pred})
    mae = lambda col: np.mean(np.abs(result['y_true'] - result[col]))
    print(f"MAE наивный: {mae('y_naive'):.2f}   MAE RNN: {mae('y_rnn'):.2f}"
          f"   MASE = {mae('y_rnn') / mae('y_naive'):.2f}")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
