#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Случайный лес: OOB-оценка и важность признаков
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURES = ['tenure', 'monthly_charge', 'support_calls', 'noise']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=42)
    rf.fit(train[FEATURES], train['churn'])

    # OOB: для каждой train-строки голосуют только деревья, в чью бутстрэп-выборку она не попала
    oob_pred = rf.oob_decision_function_.argmax(axis=1)
    assert np.isclose(rf.oob_score_, np.mean(rf.classes_[oob_pred] == train['churn'].to_numpy()))

    test_acc = accuracy_score(test['churn'], rf.predict(test[FEATURES]))
    assert np.isclose(rf.feature_importances_.sum(), 1)
    print(f"OOB = {rf.oob_score_:.3f}, test accuracy = {test_acc:.3f}")

    rows = [('oob_score', rf.oob_score_), ('test_accuracy', test_acc)]
    rows += [(f'importance_{f}', imp) for f, imp in zip(FEATURES, rf.feature_importances_)]
    return pd.DataFrame(rows, columns=['metric', 'value']).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
