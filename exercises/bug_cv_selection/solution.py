#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: отбор признаков до кросс-валидации
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv('input.csv')
    X = df.drop(columns=['target'])
    y = df['target']

    # ИСПРАВЛЕНО: отбор признаков — шаг пайплайна. cross_val_score обучает весь Pipeline
    # заново на train-части каждого фолда, и SelectKBest не видит y тестовой части.
    # Было: selector.fit_transform(X, y) на всех 120 строках до кросс-валидации.
    pipe = Pipeline([
        ('select', SelectKBest(f_classif, k=15)),
        ('model', LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f'Честная средняя accuracy: {scores.mean():.4f}')

    return pd.DataFrame({'fold': range(1, len(scores) + 1), 'accuracy': scores}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
