#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: k ближайших соседей вручную
"""

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FEATURES = ['income', 'debt_ratio']
K = 5


def knn_proba(X_train, y_train, X_test, k=K):
    # Евклидовы расстояния всех test до всех train: матрица n_test × n_train
    dist = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(dist, axis=1)
    # Проверка условия: k-й и (k+1)-й соседи не на одинаковом расстоянии
    d_sorted = np.take_along_axis(dist, order, axis=1)
    assert np.all(d_sorted[:, k] - d_sorted[:, k - 1] > 1e-9)
    return y_train[order[:, :k]].mean(axis=1)


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    y_train = train['default'].to_numpy()

    scaler = StandardScaler().fit(train[FEATURES])       # только train!
    X_train = scaler.transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])

    proba = knn_proba(X_train, y_train, X_test)
    pred = (proba > 0.5).astype(int)

    # Сверка с библиотекой
    clf = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
    assert np.allclose(proba, clf.predict_proba(X_test)[:, 1])
    assert np.array_equal(pred, clf.predict(X_test))

    # Для сравнения: без масштабирования соседей выбирает почти только доход
    raw = knn_proba(train[FEATURES].to_numpy(float), y_train, test[FEATURES].to_numpy(float))
    acc = lambda p: np.mean((p > 0.5) == test['default'].to_numpy())
    print(f"accuracy со StandardScaler: {acc(proba):.3f}   без масштабирования: {acc(raw):.3f}")
    print("предсказаний изменилось:", int(((raw > 0.5) != (proba > 0.5)).sum()))

    return pd.DataFrame({'id': test['id'], 'proba': proba.round(4), 'pred': pred})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
