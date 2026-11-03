#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: MDI против permutation importance вручную
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FEATURES = ['x_strong', 'x_strong_copy', 'x_weak', 'id_like', 'noise']


def main():
    df = pd.read_csv('input.csv')
    perms = pd.read_csv('permutations.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    X_test, y_test = test[FEATURES].reset_index(drop=True), test['y'].to_numpy()

    rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(train[FEATURES], train['y'])
    base = r2_score(y_test, rf.predict(X_test))
    print(f"R² на test: {base:.3f}")

    rows = []
    for j, col in enumerate(FEATURES):
        drops = []
        for rep in perms.columns:
            X_perm = X_test.copy()
            X_perm[col] = X_test[col].to_numpy()[perms[rep].to_numpy()]   # перемешали один столбец
            drops.append(base - r2_score(y_test, rf.predict(X_perm)))
        rows.append({'feature': col, 'mdi': rf.feature_importances_[j],
                     'perm_mean': np.mean(drops), 'perm_std': np.std(drops)})

    # Если убрать «близнеца», важность x_strong резко вырастет
    solo = [f for f in FEATURES if f != 'x_strong_copy']
    rf2 = RandomForestRegressor(n_estimators=100, random_state=0).fit(train[solo], train['y'])
    Xs = test[solo].reset_index(drop=True)
    X_perm = Xs.copy()
    X_perm['x_strong'] = Xs['x_strong'].to_numpy()[perms['rep_1'].to_numpy()]
    drop = r2_score(y_test, rf2.predict(Xs)) - r2_score(y_test, rf2.predict(X_perm))
    print(f"Без x_strong_copy падение R² при перемешивании x_strong: {drop:.3f}")

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
