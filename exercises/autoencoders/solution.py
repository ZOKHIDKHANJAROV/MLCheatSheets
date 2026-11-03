#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Линейный автоэнкодер (≈ PCA) и поиск аномалий по ошибке восстановления
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ['f1', 'f2', 'f3', 'f4']


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    scaler = StandardScaler().fit(train[FEATURES])       # параметры — только по train
    Z_train = scaler.transform(train[FEATURES])
    Z_test = scaler.transform(test[FEATURES])

    # «Обучение» линейного AE: оптимальное подпространство = 2 первые главные компоненты
    _, _, Vt = np.linalg.svd(Z_train, full_matrices=False)
    W = Vt[:2].T                                          # энкодер 4 → 2; декодер — Wᵀ

    def recon_error(Z):
        Z_hat = (Z @ W) @ W.T                             # x → z → x̂
        return ((Z - Z_hat) ** 2).mean(axis=1)

    # Сверка с PCA из sklearn
    pca = PCA(n_components=2).fit(Z_train)
    assert np.allclose(pca.inverse_transform(pca.transform(Z_test)), Z_test @ W @ W.T)

    err_train, err_test = recon_error(Z_train), recon_error(Z_test)
    threshold = np.percentile(err_train, 95)
    assert np.min(np.abs(err_test - threshold)) > 1e-3   # нет объектов «на пороге»
    print(f"порог (95-й перцентиль train): {threshold:.4f}")

    return pd.DataFrame({'id': test['id'].to_numpy(), 'recon_error': err_test.round(4),
                         'is_anomaly': (err_test > threshold).astype(int)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
