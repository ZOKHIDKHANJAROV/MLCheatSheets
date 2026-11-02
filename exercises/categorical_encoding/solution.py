#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Кодирование категориальных признаков
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

SIZE_ORDER = ['S', 'M', 'L']   # порядок по смыслу, а не по алфавиту


def main():
    df = pd.read_csv('input.csv')
    result = pd.DataFrame()

    # 1. Порядковое кодирование: явный словарь S→0, M→1, L→2
    result['size_encoded'] = df['size'].map({s: i for i, s in enumerate(SIZE_ORDER)})
    # Сверка: OrdinalEncoder с заданным порядком категорий
    ordinal = OrdinalEncoder(categories=[SIZE_ORDER]).fit_transform(df[['size']])[:, 0]
    assert (ordinal == result['size_encoded']).all()

    # 2. One-hot: по одной бинарной колонке на категорию (get_dummies сортирует по алфавиту)
    dummies = pd.get_dummies(df['category'], prefix='category', dtype=int)
    result = pd.concat([result, dummies], axis=1)
    onehot = OneHotEncoder(sparse_output=False).fit_transform(df[['category']])
    assert (onehot == dummies.to_numpy()).all()

    # 3. Target encoding: средняя цена своего цвета в каждой строке
    means = df.groupby('color')['price'].mean()
    print("Средняя цена по цвету:", means.round(2).to_dict())
    result['color_target_encoded'] = df['color'].map(means).round(2)
    assert (result['color_target_encoded']
            == df.groupby('color')['price'].transform('mean').round(2)).all()

    result['price'] = df['price']
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
