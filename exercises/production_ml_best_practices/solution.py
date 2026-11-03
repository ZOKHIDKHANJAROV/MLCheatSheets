#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Задержка инференса: перцентили, ошибки и проверка SLA
"""

import pandas as pd
import numpy as np

SLA_P95, SLA_P99, SLA_ERR = 150, 250, 0.01


def main():
    df = pd.read_csv('input.csv')
    rows = []
    for ver, g in df.groupby('model_version'):
        lat = g['latency_ms'].to_numpy()
        n = len(g)
        n_err = int((g['status'] != 'ok').sum())
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        error_rate = n_err / n
        sla_ok = int(p95 <= SLA_P95 and p99 <= SLA_P99 and error_rate <= SLA_ERR)
        rows.append({'model_version': ver, 'n': n, 'error_rate': error_rate, 'mean_ms': lat.mean(),
                     'p50': p50, 'p95': p95, 'p99': p99, 'sla_ok': sla_ok,
                     'budget_left': SLA_ERR * n - n_err})
        # сверка с pandas
        assert np.isclose(g['latency_ms'].quantile(0.99), p99)

    result = pd.DataFrame(rows)
    print(result.round(1).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
