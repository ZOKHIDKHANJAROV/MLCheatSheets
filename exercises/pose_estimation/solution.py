#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Оценка позы: OKS и PCK по ключевым точкам COCO
"""

import pandas as pd
import numpy as np

PCK_ALPHA = 0.1


def main():
    kp = pd.read_csv('input.csv').merge(pd.read_csv('sigmas.csv')[['kp_id', 'sigma']], on='kp_id')
    persons = pd.read_csv('persons.csv').set_index('person_id')

    rows = []
    for pid, g in kp.groupby('person_id', sort=True):
        g = g[g['v'] > 0]                                    # только размеченные точки
        p = persons.loc[pid]
        d2 = (g['x_pred'] - g['x_gt']) ** 2 + (g['y_pred'] - g['y_gt']) ** 2
        d = np.sqrt(d2)
        thr = PCK_ALPHA * max(p['bbox_w'], p['bbox_h'])
        assert np.min(np.abs(d - thr)) > 1e-3                # нет точек ровно на пороге PCK
        k2 = (2 * g['sigma']) ** 2
        oks = np.exp(-d2 / (2 * p['area'] * k2)).mean()      # гауссова близость с учётом размера и σ
        rows.append({'person_id': pid, 'n_labeled': len(g), 'mean_error': d.mean(),
                     'pck': (d < thr).mean(), 'oks': oks})

    result = pd.DataFrame(rows)
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
