#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Отбор признаков 5: важность признаков в случайном лесе
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']

    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Важности:", importances.round(4).to_dict())

    # Важность леса — среднее важностей отдельных деревьев (у каждого сумма = 1)
    per_tree = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    assert np.allclose(per_tree, importances) and np.isclose(importances.sum(), 1)

    top = importances.nlargest(4)
    return pd.DataFrame({'feature': top.index})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
