#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Обработка выбросов: метод IQR
"""

import pandas as pd
import numpy as np

FEATURES = ['feature1', 'feature2', 'feature3']


def main():
    df = pd.read_csv('input.csv')
    keep = pd.Series(True, index=df.index)

    for col in FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        assert np.isclose(q1, np.percentile(df[col], 25)) and np.isclose(q3, np.percentile(df[col], 75))
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inside = df[col].between(low, high)          # границы включительно
        print(f"{col}: Q1={q1}, Q3={q3}, границы [{low}, {high}], выбросы id={df.loc[~inside, 'id'].tolist()}")
        keep &= inside

    # Для сравнения: правило «3 сигмы» здесь не срабатывает — выброс сам раздувает σ
    f1 = df['feature1']
    print(f"feature1: mean + 3·std = {f1.mean() + 3 * f1.std(ddof=0):.1f} — значение 100 под порогом")

    return df.loc[keep, ['id']].sort_values('id').reset_index(drop=True)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
