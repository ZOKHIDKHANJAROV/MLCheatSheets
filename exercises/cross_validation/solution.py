#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Кросс-валидация
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


def main():
    df = pd.read_csv('input.csv')
    X, y = df[['x1', 'x2', 'x3', 'x4']], df['y']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Вручную: обучаем на 4 фолдах, проверяем на пятом
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(r2_score(y.iloc[val_idx], model.predict(X.iloc[val_idx])))

    # То же самое одной строкой
    assert np.allclose(scores, cross_val_score(Ridge(alpha=1.0), X, y, cv=kf, scoring='r2'))
    print(f"R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    return pd.DataFrame({'fold': range(1, 6), 'r2': np.round(scores, 4)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
