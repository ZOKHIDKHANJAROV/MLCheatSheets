#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Метрики классификации
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def main():
    df = pd.read_csv('input.csv')
    y, y_pred, y_score = df['y_true'], df['y_pred'], df['y_score']

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(f"Матрица ошибок: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert np.isclose(precision, precision_score(y, y_pred))
    assert np.isclose(recall, recall_score(y, y_pred))

    return pd.DataFrame([{
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_score),  # по вероятностям!
    }]).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
