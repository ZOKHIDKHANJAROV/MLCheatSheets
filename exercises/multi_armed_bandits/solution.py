#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Многорукие бандиты: ε-greedy, UCB1 и Thompson на одном потоке наград
"""

import pandas as pd
import numpy as np

MU = np.array([0.10, 0.15, 0.22, 0.18])
K, EPS = 4, 0.1


def run(choose, df, rng=None):
    R = df[['r0', 'r1', 'r2', 'r3']].to_numpy()
    n, s = np.zeros(K), np.zeros(K)
    regret, total = 0.0, 0
    for i, row in enumerate(df.itertuples()):
        t = i + 1
        a = choose(t, n, s, row, rng)
        s[a] += R[i, a]
        n[a] += 1
        total += R[i, a]
        regret += MU.max() - MU[a]
    return total, regret, n


def eps_greedy(t, n, s, row, rng):
    if t <= K:
        return t - 1                                   # стартовый раунд
    if row.u < EPS:
        return int(row.rand_arm)                       # исследуем
    return int(np.argmax(s / n))                       # используем


def ucb1(t, n, s, row, rng):
    if t <= K:
        return t - 1
    return int(np.argmax(s / n + np.sqrt(2 * np.log(t) / n)))


def thompson(t, n, s, row, rng):
    return int(np.argmax(rng.beta(1 + s, 1 + n - s)))   # сэмпл из апостериорного Beta


def main():
    df = pd.read_csv('input.csv')
    rows = []
    for name, f, rng in [('eps_greedy', eps_greedy, None), ('ucb1', ucb1, None),
                         ('thompson', thompson, np.random.default_rng(2026))]:
        total, regret, n = run(f, df, rng)
        rows.append({'strategy': name, 'total_reward': int(total), 'regret': regret,
                     'best_share': n[2] / len(df), **{f'pulls_{k}': int(n[k]) for k in range(K)}})
    result = pd.DataFrame(rows)
    print(result.round(3).to_string(index=False))
    print('случайный выбор дал бы regret ≈', round(len(df) * (MU.max() - MU.mean()), 1))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
