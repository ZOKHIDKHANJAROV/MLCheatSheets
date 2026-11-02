#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: масштабирование до разбиения
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['income', 'debt', 'age', 'credit_years']


def main():
    df = pd.read_csv('input.csv')
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    # ИСПРАВЛЕНО: SimpleImputer и StandardScaler обучаются (fit) только на train —
    # внутри Pipeline это происходит автоматически при model.fit(train).
    # Было: fit_transform на всех 480 строках, т.е. средние/std включали тестовые заявки.
    model = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        LogisticRegression(C=0.1),
    )
    model.fit(train[FEATURES], train['default'])

    # ИСПРАВЛЕНО: к тесту применяем только transform с параметрами, выученными на train
    proba = model.predict_proba(test[FEATURES])[:, 1]
    print(f"ROC-AUC на тесте: {roc_auc_score(test['default'], proba):.4f}")

    # Проверка «по одной заявке»: прогноз не зависит от остальных строк теста
    single = model.predict_proba(test[FEATURES].iloc[[0]])[:, 1][0]
    assert np.isclose(single, proba[0])

    return pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
