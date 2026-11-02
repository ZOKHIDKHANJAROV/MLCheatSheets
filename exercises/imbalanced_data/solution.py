#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Балансировка классов: random oversampling
"""

import pandas as pd
import numpy as np

from sklearn.utils import resample


def main():
    df = pd.read_csv('input.csv')
    counts = df['class'].value_counts()
    print("Объектов по классам:", counts.to_dict())

    n_extra = counts[0] - counts[1]
    minority_ids = df.loc[df['class'] == 1, 'id'].to_numpy()

    # Случайный выбор с возвращением (воспроизводимо благодаря seed)
    np.random.seed(42)
    picked = np.random.choice(minority_ids, size=n_extra, replace=True)
    print("Выбраны id:", picked)
    # sklearn.utils.resample использует тот же генератор (RandomState) — результат совпадает
    assert (picked == resample(minority_ids, replace=True, n_samples=n_extra, random_state=42)).all()

    # Копии строк в порядке выборки, с новыми id
    copies = df.set_index('id').loc[picked].reset_index()
    copies['id'] = np.arange(df['id'].max() + 1, df['id'].max() + 1 + n_extra)

    result = pd.concat([df, copies], ignore_index=True).sort_values('id').reset_index(drop=True)
    print("После балансировки:", result['class'].value_counts().to_dict())
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
