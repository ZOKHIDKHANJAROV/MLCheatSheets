#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Style transfer: матрица Грама, content loss и style loss
"""

import pandas as pd
import numpy as np

ALPHA, BETA = 1.0, 10.0
CANDIDATES = ['copy', 'style_shuffled', 'blend', 'noise']


def gram(F):
    C, H, W = F.shape
    M = F.reshape(C, H * W)             # каналы × позиции
    return M @ M.T / (C * H * W)        # сумма по позициям — расположение «забывается»


def main():
    df = pd.read_csv('input.csv').sort_values(['map', 'channel', 'row', 'col'])
    maps = {name: g['value'].to_numpy().reshape(3, 4, 4) for name, g in df.groupby('map')}
    G_style = gram(maps['style'])

    rows = []
    for name in CANDIDATES:
        F = maps[name]
        lc = np.sum((F - maps['content']) ** 2)
        ls = np.sum((gram(F) - G_style) ** 2)
        rows.append({'candidate': name, 'content_loss': lc, 'style_loss': ls,
                     'total_loss': ALPHA * lc + BETA * ls})

    result = pd.DataFrame(rows)
    assert np.isclose(result.loc[1, 'style_loss'], 0)   # перестановка позиций не меняет Грама
    print("лучший кандидат:", result.loc[result['total_loss'].idxmin(), 'candidate'])
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
