#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Метрики справедливости и выравнивание TPR порогом
"""

import pandas as pd
import numpy as np

def rates(pred, y, mask):
    return pred[mask].mean(), pred[mask & (y == 1)].mean(), pred[mask & (y == 0)].mean()


def evaluate(df, thr0, thr1, name):
    y, g, s = df['y_true'].to_numpy(), df['group'].to_numpy(), df['score'].to_numpy()
    thr = np.where(g == 0, thr0, thr1)
    pred = (s >= thr).astype(int)
    sr0, tpr0, fpr0 = rates(pred, y, g == 0)
    sr1, tpr1, fpr1 = rates(pred, y, g == 1)
    return {'scenario': name, 'thr_0': thr0, 'thr_1': thr1, 'sr_0': sr0, 'sr_1': sr1,
            'tpr_0': tpr0, 'tpr_1': tpr1, 'fpr_0': fpr0, 'fpr_1': fpr1,
            'dp_diff': sr1 - sr0, 'di_ratio': sr0 / sr1,
            'eo_gap': max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)), 'accuracy': (pred == y).mean()}


def main():
    df = pd.read_csv('input.csv')
    base = evaluate(df, 0.5, 0.5, 'base')

    # Порог группы 0, выравнивающий TPR с группой 1 (при равенстве — больший порог)
    y0 = df.loc[df['group'] == 0, 'y_true'].to_numpy()
    s0 = df.loc[df['group'] == 0, 'score'].to_numpy()
    best_t, best_gap = None, np.inf
    for t in np.round(np.arange(70, 19, -1) / 100, 2):          # от большего к меньшему
        gap = abs((s0[y0 == 1] >= t).mean() - base['tpr_1'])
        if gap < best_gap - 1e-12:
            best_t, best_gap = t, gap
    adjusted = evaluate(df, float(best_t), 0.5, 'adjusted')

    result = pd.DataFrame([base, adjusted])
    print(result[['scenario', 'thr_0', 'di_ratio', 'eo_gap', 'accuracy']].round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
