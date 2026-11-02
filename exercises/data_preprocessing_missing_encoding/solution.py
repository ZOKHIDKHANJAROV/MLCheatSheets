#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Обработка пропущенных значений и кодирование
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


def main():
    df = pd.read_csv('input.csv').drop(columns='name')
    print("Пропуски по колонкам:", df.isna().sum().to_dict())

    # Числовые колонки: среднее по известным значениям (pandas пропускает NaN сам)
    means = {col: df[col].mean() for col in ['age', 'salary', 'experience']}
    print("Средние:", {k: round(float(v), 2) for k, v in means.items()})
    # Категориальная колонка: мода
    mode_city = df['city'].mode()[0]
    print("Мода city:", mode_city)

    filled = df.fillna({**means, 'city': mode_city})

    # Сверка с SimpleImputer: та же логика «из коробки»
    num = SimpleImputer(strategy='mean').fit_transform(df[list(means)])
    assert np.allclose(num, filled[list(means)])
    cat = SimpleImputer(strategy='most_frequent').fit_transform(df[['city']])[:, 0]
    assert (cat == filled['city']).all()

    filled['age'] = filled['age'].round().astype(int)
    filled['experience'] = filled['experience'].round().astype(int)
    filled['salary'] = filled['salary'].round(2)

    # One-hot для города (колонки по алфавиту: Moscow, SPB)
    result = pd.get_dummies(filled, columns=['city'], dtype=int)
    return result[['age', 'salary', 'experience', 'city_Moscow', 'city_SPB']]


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
