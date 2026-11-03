#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: FGSM на логистической регрессии вручную и сертифицированная устойчивость
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

EPSILONS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def main():
    df = pd.read_csv('input.csv')
    pix = [f'p{j}' for j in range(64)]
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    X, y = test[pix].to_numpy(), test['label'].to_numpy()

    clf = LogisticRegression(max_iter=2000).fit(train[pix].to_numpy(), train['label'])
    w, b = clf.coef_[0], clf.intercept_[0]
    sigmoid = lambda t: 1 / (1 + np.exp(-t))

    p = sigmoid(X @ w + b)
    assert np.allclose(p, clf.predict_proba(X)[:, 1])
    clean_pred = (X @ w + b > 0).astype(int)
    grad = (p - y)[:, None] * w                      # ∂(log-loss)/∂x для каждого объекта
    margin = np.abs(X @ w + b)
    l1 = np.abs(w).sum()

    rows = []
    for eps in EPSILONS:
        X_adv = np.clip(X + eps * np.sign(grad), 0, 1)
        X_nc = X + eps * np.sign(grad)
        p_adv = sigmoid(X_adv @ w + b)
        pred_adv = (p_adv > 0.5).astype(int)
        acc_nc = ((X_nc @ w + b > 0).astype(int) == y).mean()
        cert = ((clean_pred == y) & (margin > eps * l1)).mean()
        assert np.isclose(acc_nc, cert)              # FGSM без clip — оптимальная атака на линейную модель
        rows.append({'eps': eps, 'acc_fgsm': (pred_adv == y).mean(),
                     'logloss': log_loss(y, p_adv, labels=[0, 1]),
                     'flip_rate': (pred_adv != clean_pred).mean(),
                     'acc_noclip': acc_nc, 'acc_cert': cert})

    result = pd.DataFrame(rows)
    print(f"‖w‖₁ = {l1:.2f}; медианный «запас» |логит|/‖w‖₁ = {np.median(margin / l1):.3f}")
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
