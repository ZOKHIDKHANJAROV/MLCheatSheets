#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: утечка целевой переменной
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ИСПРАВЛЕНО: убраны collection_amount (взыскание бывает только ПОСЛЕ дефолта)
# и days_to_close (договор закрывается в конце истории кредита). В момент заявки их нет.
LEAKY = ['collection_amount', 'days_to_close']
FEATURES = ['income', 'loan_amount', 'term_months', 'age', 'late_payments_12m']


def main():
    df = pd.read_csv('input.csv')
    print(df.groupby('default')[LEAKY].mean().round(1))   # видно, что признаки «знают» ответ

    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train['default'])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    print(f"ROC-AUC на тесте (честный): {roc_auc_score(test['default'], proba):.4f}")

    return pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
