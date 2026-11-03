#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Что показывают ящик с усами и гистограмма: считаем вручную
"""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')                       # рисуем без окна
import matplotlib.pyplot as plt
from matplotlib import cbook


def box_and_hist(v):
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    whisker_low = v[v >= lo_fence].min()      # ус заканчивается на реальной точке
    whisker_high = v[v <= hi_fence].max()
    n_outliers = int(((v < whisker_low) | (v > whisker_high)).sum())
    counts, edges = np.histogram(v, bins=10)
    k = counts.argmax()
    return {'n': len(v), 'median': median, 'q1': q1, 'q3': q3, 'whisker_low': whisker_low,
            'whisker_high': whisker_high, 'n_outliers': n_outliers,
            'mode_bin_left': edges[k], 'mode_bin_count': int(counts[k])}


def main():
    df = pd.read_csv('input.csv')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for g, part in df.groupby('group'):
        v = part['value'].to_numpy()
        s = box_and_hist(v)

        # Сверка с тем, что нарисует matplotlib
        ref = cbook.boxplot_stats(v)[0]
        assert np.isclose(ref['q1'], s['q1']) and np.isclose(ref['q3'], s['q3'])
        assert np.isclose(ref['whislo'], s['whisker_low']) and np.isclose(ref['whishi'], s['whisker_high'])
        assert len(ref['fliers']) == s['n_outliers']
        counts, _, _ = axes[1].hist(v, bins=10, alpha=0.5, label=g)
        assert counts.max() == s['mode_bin_count']

        rows.append({'group': g, **s})
    axes[0].boxplot([df.loc[df['group'] == g, 'value'] for g in 'ABC'], tick_labels=list('ABC'))
    axes[1].legend()
    plt.close(fig)

    result = pd.DataFrame(rows)
    print(result.round(2).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
