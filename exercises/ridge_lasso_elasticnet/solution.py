#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Ridge, Lasso и ElasticNet: путь регуляризации
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = [0.01, 0.1, 1.0, 10.0]


def main():
    df = pd.read_csv('input.csv')
    features = [f'x{j}' for j in range(1, 21)]
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    scaler = StandardScaler().fit(train[features])          # только train
    X_train, X_test = scaler.transform(train[features]), scaler.transform(test[features])
    y_train, y_test = train['y'].to_numpy(), test['y'].to_numpy()

    models = [('linear', 0.0, LinearRegression())]
    models += [('ridge', a, Ridge(alpha=a)) for a in ALPHAS]
    models += [('lasso', a, Lasso(alpha=a, max_iter=100000)) for a in ALPHAS]
    models += [('elasticnet', a, ElasticNet(alpha=a, l1_ratio=0.5, max_iter=100000)) for a in ALPHAS]

    rows = []
    for name, alpha, model in models:
        model.fit(X_train, y_train)
        rows.append({
            'model': name, 'alpha': alpha,
            'n_nonzero': int(np.sum(np.abs(model.coef_) > 1e-6)),
            'l2_norm': np.linalg.norm(model.coef_),
            'test_r2': model.score(X_test, y_test),
        })
        if name == 'lasso' and alpha == 0.1:
            kept = [f for f, w in zip(features, model.coef_) if abs(w) > 1e-6]
            print('Lasso(alpha=0.1) оставил признаки:', kept)

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
