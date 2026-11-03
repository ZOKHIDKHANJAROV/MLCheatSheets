#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Генерация GPT: greedy, temperature, top-k и top-p
"""

import pandas as pd
import numpy as np

TEMPERATURE, TOP_K, TOP_P = 0.7, 3, 0.8


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def distribution(logits, strategy):
    if strategy in ('greedy', 'top_p'):
        p = softmax(logits)
        if strategy == 'top_p':
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            keep = order[:np.argmax(cum >= TOP_P) + 1]     # минимальное «ядро» с массой ≥ p
            q = np.zeros_like(p)
            q[keep] = p[keep]
            p = q / q.sum()
        return p
    if strategy == 'temperature':
        return softmax(logits / TEMPERATURE)
    masked = np.full_like(logits, -np.inf)                  # top-k
    top = np.argsort(-logits)[:TOP_K]
    masked[top] = logits[top]
    return softmax(masked)


def main():
    table = pd.read_csv('input.csv')
    logits = table[[f'l{j}' for j in range(8)]].to_numpy()
    vocab = table['token'].tolist()
    us = pd.read_csv('uniforms.csv')['u'].to_numpy()

    rows = []
    for strategy in ['greedy', 'temperature', 'top_k', 'top_p']:
        current = vocab.index('<s>')
        for step, u in enumerate(us, start=1):
            p = distribution(logits[current], strategy)
            if strategy == 'greedy':
                nxt = int(np.argmax(p))
            else:
                nxt = int(np.argmax(np.cumsum(p) > u))       # обратная функция распределения
            rows.append({'strategy': strategy, 'step': step, 'token': vocab[nxt], 'prob': p[nxt]})
            current = nxt
        print(f"{strategy:>11}:", ' '.join(r['token'] for r in rows[-6:]))

    # Для справки: параметры GPT-2 small ≈ 12 слоёв × 12·d² + эмбеддинги V·d
    d, layers, V = 768, 12, 50257
    print(f"GPT-2 small ≈ {(12 * layers * d * d + V * d) / 1e6:.0f} млн параметров")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
