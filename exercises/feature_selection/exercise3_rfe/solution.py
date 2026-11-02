#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 3: рекурсивное удаление признаков (RFE)
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def make_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)


def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']

    # Вручную: обучаем лес, выкидываем самый неважный признак, повторяем
    features = list(X.columns)
    while len(features) > 3:
        importances = make_model().fit(X[features], y).feature_importances_
        weakest = features[int(np.argmin(importances))]
        print(f"Удаляем {weakest}: важности {pd.Series(importances, index=features).round(3).to_dict()}")
        features.remove(weakest)

    # Сверка с RFE
    rfe = RFE(make_model(), n_features_to_select=3, step=1).fit(X, y)
    print("ranking_:", pd.Series(rfe.ranking_, index=X.columns).to_dict())
    assert list(X.columns[rfe.support_]) == features

    return pd.DataFrame({'feature': features})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
