#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Свёртка, ReLU и max pooling вручную
"""

import pandas as pd
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

KERNEL = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])


def main():
    img = pd.read_csv('input.csv').to_numpy()

    # Свёртка (кросс-корреляция): окно 3×3 скользит по изображению
    windows = sliding_window_view(img, (3, 3))                    # 5 × 5 × 3 × 3
    conv = (windows * KERNEL).sum(axis=(2, 3))
    relu = np.maximum(conv, 0)
    h = relu.shape[0] // 2 * 2                                    # отбрасываем неполные окна
    pool = relu[:h, :h].reshape(h // 2, 2, h // 2, 2).max(axis=(1, 3))

    print("свёртка:\n", conv, "\nmax pooling:\n", pool)
    rows = []
    for stage, m in [('conv', conv), ('relu', relu), ('pool', pool)]:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                rows.append({'stage': stage, 'i': i, 'j': j, 'value': int(m[i, j])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
