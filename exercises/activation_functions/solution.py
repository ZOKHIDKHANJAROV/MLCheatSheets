#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Функции активации и их производные
"""

import pandas as pd
import numpy as np

def main():
    x = pd.read_csv('input.csv')['x'].to_numpy()

    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)

    result = pd.DataFrame({
        'x': x,
        'sigmoid': sigmoid, 'd_sigmoid': sigmoid * (1 - sigmoid),   # максимум 0.25 при x = 0
        'tanh': tanh, 'd_tanh': 1 - tanh ** 2,                      # максимум 1 при x = 0
        'relu': relu, 'd_relu': (x > 0).astype(float),              # не затухает при x > 0
    })
    print("max σ′ =", result['d_sigmoid'].max().round(4), "  max tanh′ =", result['d_tanh'].max().round(4))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
