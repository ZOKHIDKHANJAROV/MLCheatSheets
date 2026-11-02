#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Мультиномиальный наивный Байес для спама
"""

import pandas as pd
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CLASSES = ['ham', 'spam']
ALPHA = 1.0


def main():
    df = pd.read_csv('input.csv')
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

    vocab = sorted({w for text in train['text'] for w in text.split()})
    index = {w: j for j, w in enumerate(vocab)}

    def counts(texts):
        M = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            for w, c in Counter(text.split()).items():
                if w in index:                       # незнакомые слова игнорируем
                    M[i, index[w]] = c
        return M

    X_train, X_test = counts(list(train['text'])), counts(list(test['text']))
    y = train['label'].to_numpy()

    log_prior = np.log([np.mean(y == c) for c in CLASSES])
    N = np.array([X_train[y == c].sum(axis=0) for c in CLASSES])          # N_wc: 2 × |V|
    log_lik = np.log((N + ALPHA) / (N.sum(axis=1, keepdims=True) + ALPHA * len(vocab)))

    joint = log_prior + X_test @ log_lik.T                                 # n_test × 2
    p_spam = 1 / (1 + np.exp(joint[:, 0] - joint[:, 1]))                   # нормировка двух экспонент
    pred = np.array(CLASSES)[joint.argmax(axis=1)]

    # Сверка со sklearn
    vec = CountVectorizer(token_pattern=r'\S+', lowercase=False)
    nb = MultinomialNB(alpha=ALPHA).fit(vec.fit_transform(train['text']), y)
    Xt = vec.transform(test['text'])
    assert list(nb.classes_) == CLASSES
    assert np.allclose(joint, nb.predict_joint_log_proba(Xt))
    assert np.allclose(p_spam, nb.predict_proba(Xt)[:, 1])
    print(f"|V| = {len(vocab)}; самые «спамные» слова:",
          [vocab[j] for j in np.argsort(log_lik[1] - log_lik[0])[::-1][:3]])

    return pd.DataFrame({'id': test['id'], 'logp_ham': joint[:, 0], 'logp_spam': joint[:, 1],
                         'p_spam': p_spam, 'pred': pred}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
