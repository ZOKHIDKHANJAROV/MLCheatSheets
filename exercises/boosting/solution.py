#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Градиентный бустинг для классификации вручную
"""

import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeRegressor

LR, N_TREES = 0.3, 50
REPORT = {0, 1, 5, 10, 20, 30, 40, 50}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    X_train, y_train = train[['x1', 'x2']].to_numpy(), train['y'].to_numpy()
    X_test, y_test = test[['x1', 'x2']].to_numpy(), test['y'].to_numpy()

    p_bar = y_train.mean()
    F_train = np.full(len(y_train), np.log(p_bar / (1 - p_bar)))   # F₀ — логарифм шансов
    F_test = np.full(len(y_test), F_train[0])

    rows = []
    for m in range(N_TREES + 1):
        if m > 0:
            residual = y_train - sigmoid(F_train)                  # антиградиент log loss
            tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_train, residual)
            F_train += LR * tree.predict(X_train)
            F_test += LR * tree.predict(X_test)
        if m in REPORT:
            p_test = sigmoid(F_test)
            rows.append({'n_trees': m,
                         'train_logloss': logloss(y_train, sigmoid(F_train)),
                         'test_logloss': logloss(y_test, p_test),
                         'test_accuracy': np.mean((p_test > 0.5) == y_test)})

    assert np.isclose(rows[-1]['test_logloss'], log_loss(y_test, sigmoid(F_test)))
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
