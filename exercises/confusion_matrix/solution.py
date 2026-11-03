#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Многоклассовая матрица ошибок вручную
"""

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ['кот', 'пёс', 'лиса']


def main():
    df = pd.read_csv('input.csv')
    index = {c: i for i, c in enumerate(LABELS)}
    t = df['y_true'].map(index).to_numpy()
    p = df['y_pred'].map(index).to_numpy()

    M = np.zeros((3, 3), dtype=int)
    np.add.at(M, (t, p), 1)                     # M[истина, прогноз] += 1
    assert np.array_equal(M, confusion_matrix(df['y_true'], df['y_pred'], labels=LABELS))
    print(pd.DataFrame(M, index=LABELS, columns=LABELS))

    total = M.sum()
    rows = []
    for k, name in enumerate(LABELS):
        tp = M[k, k]
        fp = M[:, k].sum() - tp                 # столбец: предсказали k, а это не k
        fn = M[k, :].sum() - tp                 # строка: это k, а предсказали другое
        tn = total - tp - fp - fn
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        row = {'class': name, 'support': int(M[k].sum())}
        row.update({f'share_{c}': M[k, j] / M[k].sum() for j, c in enumerate(LABELS)})
        row.update({'precision': precision, 'recall': recall, 'specificity': tn / (tn + fp),
                    'f1': 2 * precision * recall / (precision + recall)})
        rows.append(row)

    result = pd.DataFrame(rows)
    prec, rec, f1, _ = precision_recall_fscore_support(df['y_true'], df['y_pred'], labels=LABELS)
    assert np.allclose(result['precision'], prec) and np.allclose(result['f1'], f1)
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
