#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 2: взаимная информация (Mutual Information)
"""

import pandas as pd
import numpy as np

from functools import partial

from sklearn.feature_selection import SelectKBest, mutual_info_classif


def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']

    mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
    print("MI:", mi.round(4).to_dict())
    # Для сравнения — линейная связь
    print("|r| Пирсона:", X.corrwith(y).abs().round(3).to_dict())

    top = mi.nlargest(3)

    # Сверка: SelectKBest с той же функцией выбирает тот же набор
    selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=3).fit(X, y)
    assert set(X.columns[selector.get_support()]) == set(top.index)

    return pd.DataFrame({'feature': top.index})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
