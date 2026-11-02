#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Снижение размерности: PCA
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ['feature1', 'feature2', 'feature3', 'feature4']


def main():
    df = pd.read_csv('input.csv')
    X = df[FEATURES].to_numpy()

    # 1. Стандартизация (ddof=0 — как в StandardScaler)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)

    # 2. Собственные векторы ковариационной матрицы; eigh сортирует значения по возрастанию
    eigvals, eigvecs = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    print("Доля объяснённой дисперсии:", (eigvals / eigvals.sum()).round(4))

    W = eigvecs[:, :2]
    # Знак: наибольшая по модулю координата каждого вектора — положительная
    for k in range(W.shape[1]):
        if W[np.argmax(np.abs(W[:, k])), k] < 0:
            W[:, k] *= -1
    print("Нагрузки PC1:", W[:, 0].round(3), " PC2:", W[:, 1].round(3))

    # 3. Проекция
    P = Z @ W

    # Сверка с sklearn (модули — на случай старых версий с другим правилом знака)
    sk = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    assert np.allclose(np.abs(P), np.abs(sk))

    return pd.DataFrame({'id': df['id'], 'PC1': P[:, 0].round(4), 'PC2': P[:, 1].round(4)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
