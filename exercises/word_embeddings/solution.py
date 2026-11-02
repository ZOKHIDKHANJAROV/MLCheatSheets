#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Word embeddings: косинусная близость и аналогии
"""

import pandas as pd
import numpy as np

def main():
    table = pd.read_csv('input.csv')
    queries = pd.read_csv('queries.csv', keep_default_na=False)
    words = table['word'].tolist()
    E = table.drop(columns='word').to_numpy()
    E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
    vec = dict(zip(words, E))

    rows = []
    for _, q in queries.iterrows():
        if q['kind'] == 'neighbors':
            v, exclude = vec[q['a']], {q['a']}
        else:
            v = vec[q['a']] - vec[q['b']] + vec[q['c']]        # a − b + c
            exclude = {q['a'], q['b'], q['c']}
        sims = E_norm @ (v / np.linalg.norm(v))
        cand = pd.DataFrame({'word': words, 'cosine': sims.round(4)})
        cand = cand[~cand['word'].isin(exclude)]
        # Тай-брейк: при равном (округлённом) косинусе — по алфавиту
        top = cand.sort_values(['cosine', 'word'], ascending=[False, True]).head(3)
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            rows.append({'query_id': q['query_id'], 'rank': rank, 'word': r['word'], 'cosine': r['cosine']})
        print(q['kind'], q['a'], q['b'], q['c'], '→', list(top['word']))

    # Проверка: косинус через sklearn
    from sklearn.metrics.pairwise import cosine_similarity
    assert np.allclose(cosine_similarity(E)[0], E_norm @ E_norm[0])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
