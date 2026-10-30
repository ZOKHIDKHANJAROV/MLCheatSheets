#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Утечка данных: отбор признаков вне пайплайна
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv('input.csv')
    X, y = df.drop(columns='target'), df['target']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Ошибка: признаки отобраны с учётом целевой переменной ВСЕХ объектов, включая будущие валидационные
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(), X_sel, y, cv=cv).mean()

    # Правильно: отбор признаков — часть модели и обучается заново в каждом фолде
    pipe = Pipeline([('select', SelectKBest(f_classif, k=20)), ('model', LogisticRegression())])
    honest = cross_val_score(pipe, X, y, cv=cv).mean()

    print(f"с утечкой: {leaky:.3f}   честно: {honest:.3f}   (данные — чистый шум)")
    return pd.DataFrame({'method': ['leaky', 'pipeline'], 'accuracy': [leaky, honest]}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
