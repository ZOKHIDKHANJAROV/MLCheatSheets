#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Экспоненциальное сглаживание: SES и Хольт
"""

import pandas as pd
import numpy as np

ALPHA, BETA, H = 0.3, 0.2, 3


def main():
    y = pd.read_csv('input.csv')['y'].to_numpy()
    n = len(y)
    ses_l = y[0]
    holt_l, holt_b = y[0], y[1] - y[0]
    rows = []
    for t in range(1, n):
        rows.append({'t': t, 'y': y[t], 'ses': ses_l, 'holt': holt_l + holt_b})   # прогнозы до наблюдения y_t
        ses_l = ALPHA * y[t] + (1 - ALPHA) * ses_l
        new_l = ALPHA * y[t] + (1 - ALPHA) * (holt_l + holt_b)
        holt_b = BETA * (new_l - holt_l) + (1 - BETA) * holt_b
        holt_l = new_l
    for h in range(1, H + 1):
        rows.append({'t': n - 1 + h, 'y': np.nan, 'ses': ses_l, 'holt': holt_l + h * holt_b})
    result = pd.DataFrame(rows)
    fit = result[result['y'].notna()]
    print(f"MAE: SES {np.abs(fit['y'] - fit['ses']).mean():.2f}   Хольт {np.abs(fit['y'] - fit['holt']).mean():.2f}")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
