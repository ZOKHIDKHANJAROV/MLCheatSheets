#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: CatBoost вручную: ordered target statistics
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

A = 1.0   # сила сглаживания


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    prior = train['bought'].mean()

    # Ordered TS: только «прошлые» объекты в перестановке
    sums, counts, ordered = {}, {}, []
    for city, y in zip(train['city'], train['bought']):
        s, c = sums.get(city, 0.0), counts.get(city, 0)
        ordered.append((s + A * prior) / (c + A))
        sums[city], counts[city] = s + y, c + 1

    # Векторная проверка той же формулы
    prev_sum = train.groupby('city')['bought'].cumsum() - train['bought']
    prev_cnt = train.groupby('city').cumcount()
    assert np.allclose(ordered, (prev_sum + A * prior) / (prev_cnt + A))

    # Greedy TS: все train-объекты города, включая сам объект
    stats = train.groupby('city')['bought'].agg(['sum', 'count'])
    greedy = ((train['city'].map(stats['sum']) + A * prior)
              / (train['city'].map(stats['count']) + A)).to_numpy()

    # Test: статистика по всему train; новый город — prior
    test_ts = ((test['city'].map(stats['sum']).fillna(0) + A * prior)
               / (test['city'].map(stats['count']).fillna(0) + A)).to_numpy()

    y = train['bought']
    print(f"ROC-AUC кодировки на train: greedy = {roc_auc_score(y, greedy):.3f}, "
          f"ordered = {roc_auc_score(y, ordered):.3f}  ← greedy «подсматривает» в ответ")

    return pd.DataFrame({
        'id': df['id'], 'split': df['split'],
        'ts_ordered': np.concatenate([ordered, test_ts]),
        'ts_greedy': np.concatenate([greedy, test_ts]),
    }).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
