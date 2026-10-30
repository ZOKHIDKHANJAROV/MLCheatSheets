#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эталонное решение мини-проекта: Кредитный скоринг с ценой ошибок
Запуск из папки проекта: python solution.py → solution_predictions.csv
"""

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

NUM = ['income', 'debt_ratio', 'age', 'credit_history_years', 'late_payments', 'card_utilization']
C_FN, C_FP = 10, 1


def main():
    train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
    prep = ColumnTransformer([('cat', OrdinalEncoder(), ['purpose'])], remainder='passthrough')
    gb = Pipeline([('prep', prep), ('gb', HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, categorical_features=[0], random_state=0))])
    # Калибровка вероятностей по кросс-валидации — порог из цены ошибок работает только для честных вероятностей
    model = CalibratedClassifierCV(gb, method='isotonic', cv=5).fit(train[['purpose'] + NUM], train['default'])

    threshold = C_FP / (C_FP + C_FN)        # байесовски оптимальный порог: отказать, если p·C_FN > (1 − p)·C_FP
    proba = model.predict_proba(test[['purpose'] + NUM])[:, 1]
    print(f"порог {threshold:.3f}; доля отказов на test: {(proba >= threshold).mean():.1%}")
    return pd.DataFrame({'id': test['id'], 'default_pred': (proba >= threshold).astype(int)})


if __name__ == "__main__":
    result = main()
    result.to_csv('solution_predictions.csv', index=False)
    print("✅ Прогнозы сохранены в solution_predictions.csv")
