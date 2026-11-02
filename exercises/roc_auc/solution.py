#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: ROC-кривая и AUC вручную
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve


def main():
    df = pd.read_csv('input.csv')
    y, score = df['y_true'].to_numpy(), df['y_score'].to_numpy()
    P, N = (y == 1).sum(), (y == 0).sum()

    thresholds = np.sort(np.unique(score))[::-1]
    tp = np.array([((score >= t) & (y == 1)).sum() for t in thresholds])
    fp = np.array([((score >= t) & (y == 0)).sum() for t in thresholds])
    tpr, fpr = tp / P, fp / N

    # Трапеции от точки (0, 0): при совпадающих оценках сегмент идёт по диагонали
    fpr0, tpr0 = np.r_[0, fpr], np.r_[0, tpr]
    auc_cum = np.cumsum(np.diff(fpr0) * (tpr0[1:] + tpr0[:-1]) / 2)

    # Сверка со sklearn: первая точка (inf, 0, 0) — служебная
    fpr_sk, tpr_sk, thr_sk = roc_curve(y, score, drop_intermediate=False)
    assert np.isinf(thr_sk[0])
    assert np.allclose(thr_sk[1:], thresholds) and np.allclose(tpr_sk[1:], tpr) and np.allclose(fpr_sk[1:], fpr)
    assert np.isclose(auc_cum[-1], roc_auc_score(y, score))
    print(f"P={P}, N={N}, ROC-AUC = {auc_cum[-1]:.4f}")

    return pd.DataFrame({'threshold': thresholds, 'tp': tp, 'fp': fp, 'tpr': tpr, 'fpr': fpr,
                         'auc_cum': auc_cum}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
