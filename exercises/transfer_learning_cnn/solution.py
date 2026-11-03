#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Transfer learning: замороженный backbone и новая голова
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = [f'e{k}' for k in range(1, 9)]


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    scaler = StandardScaler().fit(train[FEATURES])           # статистики — только по train
    head = LogisticRegression(C=1.0, max_iter=5000, tol=1e-8)
    head.fit(scaler.transform(train[FEATURES]), train['label'])

    proba = head.predict_proba(scaler.transform(test[FEATURES]))
    n_trainable = head.coef_.size + head.intercept_.size
    acc = (head.classes_[proba.argmax(axis=1)] == test['label'].to_numpy()).mean()
    print(f"обучаемых параметров головы: {n_trainable}; accuracy на test: {acc:.3f}")

    result = pd.DataFrame({'id': test['id'].to_numpy(), 'pred': head.classes_[proba.argmax(axis=1)]})
    for k, cls in enumerate(head.classes_):
        result[f'p_{cls}'] = proba[:, k].round(4)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
