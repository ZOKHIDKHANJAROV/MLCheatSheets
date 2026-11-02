#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Найди ошибку: метрики классификации
"""

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def main():
    df = pd.read_csv('input.csv')
    y_true = df['y_true']
    y_pred = (df['y_score'] >= 0.5).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    accuracy = (y_pred == y_true).mean()
    # ИСПРАВЛЕНО: precision = TP / (TP + FP) — доля верных среди предсказанных «1»
    precision = tp / (tp + fp)
    # ИСПРАВЛЕНО: recall = TP / (TP + FN) — доля найденных среди настоящих «1»
    recall = tp / (tp + fn)
    # ИСПРАВЛЕНО: F1 для положительного класса — average='binary' (по умолчанию),
    # а не 'macro' (среднее F1 класса 0 и класса 1)
    f1 = f1_score(y_true, y_pred)
    # ИСПРАВЛЕНО: ROC-AUC считается по вероятностям y_score, а не по меткам y_pred
    roc_auc = roc_auc_score(y_true, df['y_score'])

    assert np.isclose(precision, precision_score(y_true, y_pred))
    assert np.isclose(recall, recall_score(y_true, y_pred))
    assert np.isclose(f1, 2 * precision * recall / (precision + recall))

    return pd.DataFrame([{
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
    }]).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
