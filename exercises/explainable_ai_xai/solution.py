#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Permutation importance вручную и сравнение с MDI
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

FEATURES = ['x1', 'x2', 'x3', 'x4']
N_REPEATS = 5


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    X_tr, y_tr = train[FEATURES].to_numpy(), train['y'].to_numpy()
    X_te, y_te = test[FEATURES].to_numpy(), test['y'].to_numpy()

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0).fit(X_tr, y_tr)
    base = r2_score(y_te, model.predict(X_te))
    print(f"R² на test: {base:.4f}")

    rng = np.random.default_rng(42)
    drops = np.zeros((len(FEATURES), N_REPEATS))
    for j in range(len(FEATURES)):
        for r in range(N_REPEATS):
            perm = rng.permutation(len(X_te))
            Xp = X_te.copy()
            Xp[:, j] = X_te[perm, j]                 # перемешан только признак j
            drops[j, r] = base - r2_score(y_te, model.predict(Xp))

    perm_mean = drops.mean(axis=1)
    result = pd.DataFrame({'feature': FEATURES, 'mdi': model.feature_importances_,
                           'perm_mean': perm_mean, 'perm_std': drops.std(axis=1)})
    result['rank_perm'] = result['perm_mean'].rank(ascending=False, method='first').astype(int)

    # Для сравнения — библиотечная реализация (другие перестановки, близкие значения)
    lib = permutation_importance(model, X_te, y_te, n_repeats=N_REPEATS, random_state=0)
    print('sklearn permutation_importance:', lib.importances_mean.round(3))
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
