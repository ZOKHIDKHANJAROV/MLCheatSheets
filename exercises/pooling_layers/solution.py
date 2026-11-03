#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Max и average pooling со stride и padding
"""

import pandas as pd
import numpy as np

def pool(x, mode, k, s, p):
    fill = -np.inf if mode == 'max' else 0.0
    xp = np.pad(x.astype(float), p, constant_values=fill)
    H = (x.shape[0] + 2 * p - k) // s + 1
    W = (x.shape[1] + 2 * p - k) // s + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            win = xp[i * s:i * s + k, j * s:j * s + k]
            out[i, j] = win.max() if mode == 'max' else win.sum() / (k * k)   # делим на k², вкл. паддинг
    return out


def main():
    px = pd.read_csv('input.csv')
    x = px.pivot(index='row', columns='col', values='value').to_numpy()
    configs = pd.read_csv('configs.csv')

    rows = []
    for c in configs.itertuples():
        if c.mode == 'global_avg':
            out = np.array([[x.mean()]])
        elif c.mode == 'global_max':
            out = np.array([[x.max()]])
        else:
            out = pool(x, c.mode, c.kernel, c.stride, c.padding)
        print(f"config {c.config_id} ({c.mode}): {x.shape} → {out.shape}")
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                rows.append({'config_id': c.config_id, 'out_h': out.shape[0], 'out_w': out.shape[1],
                             'row': i, 'col': j, 'value': round(float(out[i, j]), 4)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
