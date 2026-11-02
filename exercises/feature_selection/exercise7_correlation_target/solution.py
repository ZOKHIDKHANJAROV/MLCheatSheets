#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 7: корреляция с целевой переменной
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']

    # Вручную по формуле Пирсона
    dx, dy = X - X.mean(), y - y.mean()
    r = (dx.mul(dy, axis=0)).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    assert np.allclose(r, df.corr()['target'].drop('target'))
    print("r с target:", r.round(4).to_dict())

    top = r.abs().sort_values(ascending=False).head(3)
    return pd.DataFrame({'feature': top.index})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
