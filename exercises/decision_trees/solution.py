#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Дерево решений: одобрение кредита
"""

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ['income', 'debt', 'credit_years', 'age']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(train[FEATURES], train['approved'])
    print(export_text(tree, feature_names=FEATURES))
    print("Важность признаков:", dict(zip(FEATURES, tree.feature_importances_.round(3))))

    # Для сравнения: дерево без ограничения глубины переобучается
    deep = DecisionTreeClassifier(random_state=42).fit(train[FEATURES], train['approved'])
    for name, m in [('max_depth=3', tree), ('без ограничения', deep)]:
        print(f"{name:>16}: train={accuracy_score(train['approved'], m.predict(train[FEATURES])):.3f}"
              f"  test={accuracy_score(test['approved'], m.predict(test[FEATURES])):.3f}")

    return pd.DataFrame({'id': test['id'], 'approved_pred': tree.predict(test[FEATURES])})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
