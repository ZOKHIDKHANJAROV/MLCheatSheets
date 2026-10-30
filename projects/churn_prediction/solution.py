#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эталонное решение мини-проекта: Отток клиентов телеком-оператора
Запуск из папки проекта: python solution.py → solution_predictions.csv
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

NUM = ['tenure', 'monthly_charge', 'total_charges', 'support_calls', 'age', 'satisfaction']
CAT = ['contract', 'internet', 'payment', 'region']


def main():
    train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
    for df in (train, test):
        df['satisfaction_missing'] = df['satisfaction'].isna().astype(int)   # пропуск в опросе информативен
        df['charge_per_month'] = df['total_charges'] / df['tenure']
    num = NUM + ['satisfaction_missing', 'charge_per_month']

    # Градиентный бустинг сам обрабатывает пропуски; категории кодируем порядковыми номерами
    prep = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT)],
                             remainder='passthrough')
    model = Pipeline([('prep', prep), ('gb', HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        categorical_features=list(range(len(CAT))), random_state=0))])

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(model, train[CAT + num], train['churn'], cv=cv, scoring='roc_auc')
    print(f"ROC-AUC на CV: {auc.mean():.3f} ± {auc.std():.3f}")

    model.fit(train[CAT + num], train['churn'])
    return pd.DataFrame({'id': test['id'], 'churn_proba': model.predict_proba(test[CAT + num])[:, 1].round(5)})


if __name__ == "__main__":
    result = main()
    result.to_csv('solution_predictions.csv', index=False)
    print("✅ Прогнозы сохранены в solution_predictions.csv")
