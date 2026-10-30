#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эталонное решение мини-проекта: Распознавание рукописных цифр нейросетью
Запуск из папки проекта: python solution.py → solution_predictions.csv
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

COLS = [f'p{i}' for i in range(64)]


def main():
    train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
    X, Xt = train[COLS] / 16, test[COLS] / 16      # яркость 0–16 → 0–1

    # Ранняя остановка на маленькой выборке останавливает обучение слишком рано (≈0.93) —
    # вместо неё регуляризуем L2 (alpha) и оцениваем качество на отложенной части
    X_tr, X_val, y_tr, y_val = train_test_split(X, train['label'], test_size=0.2, stratify=train['label'], random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1000, random_state=0)
    print(f"accuracy на валидации: {mlp.fit(X_tr, y_tr).score(X_val, y_val):.3f}")

    mlp.fit(X, train['label'])          # финальная модель на всех обучающих данных
    return pd.DataFrame({'id': test['id'], 'label': mlp.predict(Xt)})


if __name__ == "__main__":
    result = main()
    result.to_csv('solution_predictions.csv', index=False)
    print("✅ Прогнозы сохранены в solution_predictions.csv")
