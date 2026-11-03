#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: NumPy: оси, broadcasting, маски и матричное умножение
"""

import pandas as pd
import numpy as np

import json

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv('input.csv')
    params = json.load(open('weights.json', encoding='utf-8'))
    ids = df['id'].to_numpy()
    X = df[['height', 'weight', 'age']].to_numpy(dtype=float)     # (30, 3)
    w, b = np.array(params['w']), params['b']

    # 1) Стандартизация: агрегаты по оси объектов, broadcasting (30, 3) − (3,)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    assert np.allclose(Z, StandardScaler().fit_transform(X))

    # 2) Линейная модель для всех объектов сразу
    score = Z @ w + b                                             # (30,)

    # 3) Булева маска и сумма по оси признаков
    n_extreme = (np.abs(Z) > 2).sum(axis=1)

    # 4) Попарные расстояния через broadcasting: (30, 1, 3) − (1, 30, 3) → (30, 30, 3)
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)                                # себя не считаем соседом
    j = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(Z)), j]
    second = np.sort(dist, axis=1)[:, 1]
    assert np.all(second - nearest_dist > 1e-9)                   # ничьих нет

    # Сверка с sklearn: первый сосед — сам объект, второй — ближайший другой
    nn_dist, nn_idx = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    assert np.array_equal(nn_idx[:, 1], j) and np.allclose(nn_dist[:, 1], nearest_dist)

    print('объекты с экстремальными значениями:', ids[n_extreme > 0])
    return pd.DataFrame({'id': ids, 'score': score.round(4), 'n_extreme': n_extreme,
                         'nearest_id': ids[j], 'nearest_dist': nearest_dist.round(4)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
