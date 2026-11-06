#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Векторный поиск: косинус, нормализация и IVF-подобное приближение
"""

import pandas as pd
import numpy as np

K, NPROBES = 5, [1, 2, 3, 6]


def normalize(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def main():
    base = pd.read_csv('input.csv')
    queries = pd.read_csv('queries.csv')
    cents = pd.read_csv('centroids.csv')
    cols = [f'x{j}' for j in range(8)]
    X, Q, C = base[cols].to_numpy(), queries[cols].to_numpy(), cents[cols].to_numpy()
    Xn, Qn, Cn = normalize(X), normalize(Q), normalize(C)

    cos = Qn @ Xn.T                                          # косинусы: 20 × 300
    exact = np.argsort(-cos, axis=1, kind='stable')[:, :K]   # точный top-5 (позиции в базе)

    raw_best = np.argmax(Q @ X.T, axis=1)                    # лучший по сырому скалярному произведению
    differs = int((raw_best != exact[:, 0]).sum())

    cell = np.argmax(Xn @ Cn.T, axis=1)                      # ячейка каждого вектора базы
    rows = [{'name': 'top1_differs_dot_vs_cosine', 'value': differs}]
    for nprobe in NPROBES:
        recalls, scanned = [], []
        for i in range(len(Q)):
            probes = np.argsort(-(Cn @ Qn[i]), kind='stable')[:nprobe]
            cand = np.flatnonzero(np.isin(cell, probes))
            top = cand[np.argsort(-cos[i, cand], kind='stable')[:K]]
            recalls.append(len(set(top) & set(exact[i])) / K)
            scanned.append(len(cand) / len(X))
        rows.append({'name': f'recall_at_5_nprobe_{nprobe}', 'value': np.mean(recalls)})
        rows.append({'name': f'scanned_share_nprobe_{nprobe}', 'value': np.mean(scanned)})
        print(f"nprobe={nprobe}: recall@5 = {np.mean(recalls):.3f}, просмотрено {100 * np.mean(scanned):.0f}% базы")

    print(f"лучший по сырому скалярному произведению и по косинусу различаются у {differs} запросов из {len(Q)}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
