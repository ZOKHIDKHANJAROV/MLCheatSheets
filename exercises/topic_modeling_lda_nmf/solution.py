#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: NMF для тем: мультипликативные обновления Ли — Сына
"""

import pandas as pd
import numpy as np

import json
import warnings

N_ITER = 200


def main():
    df = pd.read_csv('input.csv')
    words = df.columns[1:].tolist()
    X = df[words].to_numpy(dtype=float)
    init = json.load(open('init.json', encoding='utf-8'))
    W0, H0 = np.array(init['W0']), np.array(init['H0'])

    W, H = W0.copy(), H0.copy()
    for it in range(1, N_ITER + 1):
        W *= (X @ H.T) / (W @ H @ H.T)                  # сначала W
        H *= (W.T @ X) / (W.T @ W @ H)                  # потом H — уже с новым W
        if it % 50 == 0:
            print(f"итерация {it}: ‖X − WH‖ = {np.linalg.norm(X - W @ H):.4f}")

    # Проверка: то же самое делает sklearn
    from sklearn.decomposition import NMF
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = NMF(n_components=3, init='custom', solver='mu', beta_loss='frobenius',
                    max_iter=N_ITER, tol=0)
        model.fit_transform(X, W=W0.copy(), H=H0.copy())
    assert np.allclose(model.components_, H, atol=1e-6)

    rows = []
    for k in range(H.shape[0]):
        top = np.argsort(-H[k])[:4]
        rows += [{'topic': k, 'rank': r, 'word': words[j], 'weight': H[k, j]}
                 for r, j in enumerate(top, start=1)]
        print(f"тема {k}:", ', '.join(words[j] for j in top))
    print("главная тема документов:", W.argmax(axis=1))
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
