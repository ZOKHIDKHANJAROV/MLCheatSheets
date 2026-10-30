#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Логистическая регрессия: отток клиентов
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES = ['tenure', 'monthly_charge', 'support_calls', 'age']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    # Масштабирование обучаем только на train — иначе утечка данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])

    model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, train['churn'])
    for name, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {name:>15}: {coef:+.3f}")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"Accuracy={accuracy_score(test['churn'], pred):.3f}  "
          f"ROC-AUC={roc_auc_score(test['churn'], proba):.3f}")

    return pd.DataFrame({'id': test['id'], 'churn_proba': proba.round(4), 'churn_pred': pred})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
