#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Метрики регрессии
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = pd.read_csv('input.csv')
    y, y_hat = df['y_true'], df['y_pred']

    # Вручную — чтобы понимать формулы
    errors = y - y_hat
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)
    mape = 100 * np.mean(np.abs(errors) / np.abs(y))

    # Сверка с библиотекой
    assert np.isclose(mae, mean_absolute_error(y, y_hat))
    assert np.isclose(mse, mean_squared_error(y, y_hat))
    assert np.isclose(r2, r2_score(y, y_hat))

    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  (RMSE ≥ MAE: большие ошибки штрафуются сильнее)")
    return pd.DataFrame([{
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape,
    }]).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
