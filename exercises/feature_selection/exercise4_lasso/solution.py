#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 4: Lasso (L1-регуляризация)
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']
    Z = StandardScaler().fit_transform(X)

    lasso = Lasso(alpha=0.1).fit(Z, y)
    print("Коэффициенты:", pd.Series(lasso.coef_, index=X.columns).round(4).to_dict())
    selected = X.columns[lasso.coef_ != 0]

    # Чем больше alpha, тем больше нулей
    for alpha in [0.3, 1.0]:
        coef = Lasso(alpha=alpha).fit(Z, y).coef_
        print(f"alpha={alpha}: отобраны {list(X.columns[coef != 0])}")

    # Сверка: SelectFromModel с Lasso (для L1-моделей порог по умолчанию — почти ноль)
    sfm = SelectFromModel(Lasso(alpha=0.1)).fit(Z, y)
    assert list(X.columns[sfm.get_support()]) == list(selected)

    return pd.DataFrame({'feature': selected})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
