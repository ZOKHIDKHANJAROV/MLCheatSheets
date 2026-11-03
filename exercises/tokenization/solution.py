#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: BPE вручную: список слияний на маленьком корпусе
"""

import pandas as pd
import numpy as np

from collections import Counter

N_MERGES = 10


def pair_counts(vocab):
    counts = Counter()
    for symbols, freq in vocab.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def merge_word(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def tokenize(word, merges):
    symbols = tuple(word) + ('</w>',)
    for pair in merges:                     # слияния применяются в порядке обучения
        symbols = merge_word(symbols, pair)
    return symbols


def main():
    df = pd.read_csv('input.csv')
    vocab = {tuple(w) + ('</w>',): int(c) for w, c in zip(df['word'], df['count'])}
    tokens = {s for symbols in vocab for s in symbols}

    rows, merges = [], []
    for step in range(1, N_MERGES + 1):
        counts = pair_counts(vocab)
        best = min(counts, key=lambda p: (-counts[p], p))   # самая частая; тай-брейк по строкам
        vocab = {merge_word(s, best): f for s, f in vocab.items()}
        merges.append(best)
        tokens.add(best[0] + best[1])
        rows.append({'step': step, 'left': best[0], 'right': best[1], 'new_token': best[0] + best[1],
                     'freq': counts[best], 'vocab_size': len(tokens)})

    for w in ['кроты', 'ворот', 'котлета']:
        print(w, '→', ' '.join(tokenize(w, merges)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
