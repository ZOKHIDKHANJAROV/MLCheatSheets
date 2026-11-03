#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Quality gate в CI/CD: пропустить модель или остановить конвейер
"""

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

MIN_GAIN_PROD, MIN_GAIN_BASE, MAX_SEG_DROP = 0.005, 0.02, 0.01
MAX_LATENCY, MAX_SIZE = 120, 50


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def main():
    cm = pd.read_csv('input.csv')
    meta = pd.read_csv('meta.csv').set_index('run_id')
    cm['P'], cm['N'] = cm['tp'] + cm['fn'], cm['fp'] + cm['tn']
    cm['f1'] = f1(cm['tp'], cm['fp'], cm['fn'])
    seg = cm.set_index(['run_id', 'segment'])
    total = cm.groupby('run_id')[['tp', 'fp', 'fn', 'P', 'N']].sum()
    total['f1'] = f1(total['tp'], total['fp'], total['fn'])

    prod_f1 = total.loc['prod', 'f1']
    P, N = total.loc['prod', 'P'], total.loc['prod', 'N']
    base_f1 = f1(P, N, 0)                          # «всегда 1»: TP = P, FP = N, FN = 0
    # та же формула через sklearn
    y = np.r_[np.ones(P), np.zeros(N)]
    assert np.isclose(base_f1, f1_score(y, np.ones_like(y)))
    print(f"F1 prod = {prod_f1:.4f}, F1 бейзлайна = {base_f1:.4f}")

    rows = []
    for run in [r for r in total.index if r != 'prod']:
        s, p = seg.loc[run], seg.loc['prod']
        checks = {
            'smoke': meta.loc[run, 'smoke_test'] == 'pass',
            'same_data': bool(((s['P'] == p['P']) & (s['N'] == p['N'])).all()),
            'vs_prod': total.loc[run, 'f1'] >= prod_f1 + MIN_GAIN_PROD,
            'vs_baseline': total.loc[run, 'f1'] >= base_f1 + MIN_GAIN_BASE,
            'segments': bool((s['f1'] >= p['f1'] - MAX_SEG_DROP).all()),
            'latency': meta.loc[run, 'latency_p95_ms'] <= MAX_LATENCY,
            'size': meta.loc[run, 'size_mb'] <= MAX_SIZE,
        }
        failed = ';'.join(k for k, ok in checks.items() if not ok) or 'none'
        rows.append({'run_id': run, 'f1': total.loc[run, 'f1'],
                     'delta_vs_prod': total.loc[run, 'f1'] - prod_f1,
                     'decision': 'PASS' if failed == 'none' else 'FAIL', 'failed': failed})

    result = pd.DataFrame(rows)
    passed = result[result['decision'] == 'PASS']
    result['promote'] = 0
    if len(passed):
        result.loc[passed['f1'].idxmax(), 'promote'] = 1
    print(result.to_string(index=False))
    return result.round({'f1': 4, 'delta_vs_prod': 4})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
