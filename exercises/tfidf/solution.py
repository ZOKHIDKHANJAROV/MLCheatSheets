#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: TF-IDF вручную
"""

import pandas as pd
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    df = pd.read_csv('input.csv')
    docs = [t.split() for t in df['text']]
    N = len(docs)
    vocab = sorted({w for d in docs for w in d})
    dfreq = {w: sum(w in d for d in docs) for w in vocab}
    idf = {w: np.log((1 + N) / (1 + dfreq[w])) + 1 for w in vocab}

    rows, matrix = [], []
    for doc_id, d in zip(df['doc'], docs):
        tf = Counter(d)
        vec = np.array([tf[w] * idf[w] for w in vocab])
        vec = vec / np.linalg.norm(vec)
        matrix.append(vec)
        rows += [{'doc': doc_id, 'term': w, 'tfidf': v} for w, v in zip(vocab, vec) if v > 0]

    # Сверка с библиотекой
    ref = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False).fit(df['text'])
    lib = ref.transform(df['text']).toarray()[:, [ref.vocabulary_[w] for w in vocab]]
    assert np.allclose(lib, np.array(matrix))
    print("совпадает с TfidfVectorizer")

    return pd.DataFrame(rows).sort_values(['doc', 'term']).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
