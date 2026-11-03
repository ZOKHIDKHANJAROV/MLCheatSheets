#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Мониторинг: PSI и KS по признакам и z-тест A/B с доверительным интервалом
"""

import pandas as pd
import numpy as np

from scipy import stats

FEATURES = ['income', 'session_min', 'basket']


def psi(ref, cur, bins=10, eps=1e-6):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref) + eps
    a = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    df = pd.read_csv('input.csv')
    ref, cur = df[df['sample'] == 'reference'], df[df['sample'] == 'current']
    rows = []
    for f in FEATURES:
        rows.append(('psi_' + f, psi(ref[f].to_numpy(), cur[f].to_numpy())))
    for f in FEATURES:
        rows.append(('ks_' + f, stats.ks_2samp(ref[f], cur[f]).statistic))
    for name, value in rows[:3]:
        level = 'стабильно' if value < 0.1 else 'умеренный сдвиг' if value < 0.25 else 'значимый сдвиг'
        print(f"{name}: {value:.3f} — {level}")

    ab = pd.read_csv('ab.csv')
    g = ab.groupby('group')['converted'].agg(['sum', 'count'])
    (x_a, n_a), (x_b, n_b) = g.loc['A'], g.loc['B']
    p_a, p_b = x_a / n_a, x_b / n_b
    d = p_b - p_a
    p = (x_a + x_b) / (n_a + n_b)
    z = d / np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))          # SE при H0: объединённая доля
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)  # SE для интервала: раздельная
    q = stats.norm.ppf(0.975)
    rows += [('conv_a', p_a), ('conv_b', p_b), ('diff', d), ('z', z), ('p_value', p_value),
             ('ci_low', d - q * se), ('ci_high', d + q * se)]
    print(f"A/B: {p_a:.2%} → {p_b:.2%}, z = {z:.2f}, p = {p_value:.4f}, "
          f"95% ДИ [{d - q * se:+.4f}; {d + q * se:+.4f}]")

    result = pd.DataFrame(rows, columns=['metric', 'value'])
    result['value'] = result['value'].astype(float).round(4)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
