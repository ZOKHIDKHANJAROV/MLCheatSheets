#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: LightGBM вручную: гистограммы и поиск разбиения по корзинам
"""

import pandas as pd
import numpy as np

N_BINS, LAMBDA = 8, 1.0
FEATURES = ['x1', 'x2']


def gain_formula(GL, HL, G, H):
    GR, HR = G - GL, H - HL
    return GL ** 2 / (HL + LAMBDA) + GR ** 2 / (HR + LAMBDA) - G ** 2 / (H + LAMBDA)


def main():
    df = pd.read_csv('input.csv')
    y = df['y'].to_numpy()
    g = y.mean() - y                      # градиент ½(F − y)² в точке F₀ = mean(y)
    h = np.ones_like(g)                   # гессиан
    G, H = g.sum(), h.sum()

    rows = []
    for name in FEATURES:
        x = df[name].to_numpy()
        edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
        bins = np.searchsorted(edges[1:-1], x, side='right')       # 0 … 7

        count = np.bincount(bins, minlength=N_BINS)
        grad_sum = np.bincount(bins, weights=g, minlength=N_BINS)
        hess_sum = np.bincount(bins, weights=h, minlength=N_BINS)

        GL, HL = np.cumsum(grad_sum), np.cumsum(hess_sum)          # влево — корзины 0…k
        gain = gain_formula(GL, HL, G, H)
        gain[-1] = 0.0                                             # после последней корзины делить нечего

        # Для сравнения: точный перебор всех порогов (как без гистограмм)
        order = np.argsort(x)
        exact = gain_formula(np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1], G, H)
        print(f"{name}: лучший gain по 8 корзинам = {gain.max():.2f}, "
              f"точный по {len(x) - 1} порогам = {exact.max():.2f}")

        rows += [{'feature': name, 'bin': k, 'count': int(count[k]),
                  'grad_sum': grad_sum[k], 'gain': gain[k]} for k in range(N_BINS)]

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
