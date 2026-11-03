#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Residual-блок ResNet с проекцией в skip-связи
"""

import pandas as pd
import numpy as np

import json


def conv2d(x, W, stride, pad):
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    C_out, _, k, _ = W.shape
    H = (x.shape[1] + 2 * pad - k) // stride + 1
    Wd = (x.shape[2] + 2 * pad - k) // stride + 1
    out = np.zeros((C_out, H, Wd))
    for o in range(C_out):
        for i in range(H):
            for j in range(Wd):
                out[o, i, j] = np.sum(W[o] * xp[:, i * stride:i * stride + k, j * stride:j * stride + k])
    return out


def batchnorm_eval(x, p, eps):
    g, b, m, v = (np.array(p[key])[:, None, None] for key in ['gamma', 'beta', 'running_mean', 'running_var'])
    return (x - m) / np.sqrt(v + eps) * g + b


def main():
    df = pd.read_csv('input.csv').sort_values(['channel', 'row', 'col'])
    x = df['value'].to_numpy().reshape(2, 5, 5)
    w = json.load(open('weights.json', encoding='utf-8'))
    eps = w['eps']

    out = np.maximum(0, batchnorm_eval(conv2d(x, np.array(w['conv1']), 2, 1), w['bn1'], eps))
    F = batchnorm_eval(conv2d(out, np.array(w['conv2']), 1, 1), w['bn2'], eps)       # остаток F(x)
    S = batchnorm_eval(conv2d(x, np.array(w['proj']), 2, 0), w['bn_proj'], eps)      # проекция x
    assert F.shape == S.shape == (4, 3, 3)
    y = np.maximum(0, F + S)

    n_params = sum(np.array(w[k]).size for k in ['conv1', 'conv2', 'proj']) + 3 * 2 * 4
    print(f"обучаемых параметров в блоке: {n_params}; нулевых выходов после ReLU: {(y == 0).sum()} из {y.size}")

    C, H, W = y.shape
    idx = np.indices((C, H, W)).reshape(3, -1)
    return pd.DataFrame({'channel': idx[0], 'row': idx[1], 'col': idx[2], 'residual': F.ravel(),
                         'shortcut': S.ravel(), 'out': y.ravel()}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
