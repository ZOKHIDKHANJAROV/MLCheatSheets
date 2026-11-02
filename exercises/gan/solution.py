#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Функции потерь GAN: minimax и non-saturating
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('input.csv')
    rows = []
    for batch, g in df.groupby('batch', sort=True):
        d_real = g.loc[g['kind'] == 'real', 'd_out'].to_numpy()
        d_fake = g.loc[g['kind'] == 'fake', 'd_out'].to_numpy()

        d_loss = -np.mean(np.log(d_real)) - np.mean(np.log(1 - d_fake))   # BCE(real, 1) + BCE(fake, 0)
        g_minimax = np.mean(np.log(1 - d_fake))                          # минимизирует G в исходной игре
        g_ns = -np.mean(np.log(d_fake))                                  # BCE(fake, 1)

        # d/da log(1 − σ(a)) = −σ(a);  d/da (−log σ(a)) = −(1 − σ(a))
        rows.append({'batch': batch, 'd_loss': d_loss, 'g_loss_minimax': g_minimax, 'g_loss_ns': g_ns,
                     'grad_minimax': np.mean(d_fake), 'grad_ns': np.mean(1 - d_fake)})

    result = pd.DataFrame(rows)
    # Численная проверка производной на одном значении
    a, eps = 0.3, 1e-6
    s = lambda v: 1 / (1 + np.exp(-v))
    assert np.isclose((np.log(1 - s(a + eps)) - np.log(1 - s(a - eps))) / (2 * eps), -s(a), atol=1e-6)
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
