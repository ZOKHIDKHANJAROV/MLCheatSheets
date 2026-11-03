#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Аддитивное внимание Bahdanau
"""

import pandas as pd
import numpy as np

import json


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    enc = pd.read_csv('input.csv')
    dec = pd.read_csv('decoder.csv')
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    H = enc[['h0', 'h1', 'h2']].to_numpy()             # 6 × 3 — «ключи» и «значения»

    keys = H @ w['W_k'].T                              # W_k·hⱼ для всех слов: 6 × 4, считаем один раз
    rows = []
    for step, s in zip(dec['step'], dec[['s0', 's1']].to_numpy()):
        scores = np.tanh(w['W_q'] @ s + keys) @ w['v']  # eⱼ = vᵀ·tanh(W_q·s + W_k·hⱼ)
        alpha = softmax(scores)                         # распределение по словам источника
        context = alpha @ H                             # cₜ = Σ αⱼ·hⱼ
        assert np.isclose(alpha.sum(), 1)
        row = {'step': step}
        row.update({f'a{j + 1}': alpha[j] for j in range(len(alpha))})
        row.update({f'c{k}': context[k] for k in range(H.shape[1])})
        row['top_word'] = enc['word'][int(alpha.argmax())]
        rows.append(row)
        print(f"шаг {step}: декодер смотрит в основном на «{row['top_word']}» (α = {alpha.max():.3f})")

    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
