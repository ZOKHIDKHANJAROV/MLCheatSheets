#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Обучение перцептрона Розенблатта вручную
"""

import pandas as pd
import numpy as np

LR, MAX_EPOCHS = 0.1, 100


def main():
    df = pd.read_csv('input.csv')
    X = df[['x1', 'x2']].to_numpy(dtype=float)
    y = df['y'].to_numpy()

    w, b = np.zeros(2), 0.0
    rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        n_errors = 0
        for xi, yi in zip(X, y):
            z = xi @ w + b
            # Однозначность: ненулевой z не должен быть «почти нулём» из-за округлений
            assert z == 0 or abs(z) > 1e-9
            err = yi - int(z > 0)            # ступенчатая активация
            w += LR * err * xi               # правило перцептрона
            b += LR * err
            n_errors += int(err != 0)
        acc = np.mean(((X @ w + b) > 0).astype(int) == y)
        rows.append({'epoch': epoch, 'n_errors': n_errors, 'w1': w[0], 'w2': w[1], 'b': b, 'accuracy': acc})
        if n_errors == 0:
            break
    else:
        raise RuntimeError('перцептрон не сошёлся')

    print(f"Сошёлся за {len(rows)} эпох: {w[0]:.2f}·x1 + {w[1]:.2f}·x2 + {b:.2f} = 0")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
