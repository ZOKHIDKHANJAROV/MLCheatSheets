#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: RAG: чанкинг с перекрытием, recall@k и MRR
"""

import pandas as pd
import numpy as np

CONFIGS = [(10, 0), (10, 3), (20, 0), (20, 5)]
K = 3


def chunk(words, size, overlap):
    step = size - overlap
    return [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]


def evaluate(docs, queries, size, overlap):
    chunks, owner = [], []                          # сквозная нумерация: документ 1, 2, ...
    for doc_id, text in zip(docs['doc_id'], docs['text']):
        for c in chunk(text.split(), size, overlap):
            chunks.append(set(c))
            owner.append(doc_id)
    n = len(chunks)
    df = {}
    for c in chunks:
        for w in c:
            df[w] = df.get(w, 0) + 1
    idf = lambda w: np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))

    recalls, rr = [], []
    for q in queries.itertuples():
        terms = sorted(set(q.query.split()))
        scores = [round(sum(idf(w) for w in terms if w in c), 9) for c in chunks]
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        relevant = {i for i in range(n) if owner[i] == q.doc_id and q.answer in chunks[i]}
        recalls.append(len(set(order[:K]) & relevant) / len(relevant))
        rank = next(r for r, i in enumerate(order, 1) if i in relevant)
        rr.append(1 / rank)
    return n, np.mean(recalls), np.mean(rr)


def main():
    docs = pd.read_csv('input.csv')
    queries = pd.read_csv('queries.csv')
    rows = []
    for size, overlap in CONFIGS:
        n, recall, mrr = evaluate(docs, queries, size, overlap)
        rows.append({'size': size, 'overlap': overlap, 'n_chunks': n,
                     'recall_at_3': recall, 'mrr': mrr})
        print(f"size={size:2d} overlap={overlap}: чанков {n}, recall@3 = {recall:.3f}, MRR = {mrr:.3f}")
    result = pd.DataFrame(rows)
    result[['recall_at_3', 'mrr']] = result[['recall_at_3', 'mrr']].round(4)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
