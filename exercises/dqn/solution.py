#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: DQN: мини-батч из replay buffer, цели target-сети и Double DQN
"""

import pandas as pd
import numpy as np

import json

GAMMA, SEED, BATCH = 0.99, 2024, 8


def huber(e, delta=1.0):
    return np.where(np.abs(e) < delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))


def main():
    buf = pd.read_csv('input.csv')
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    q_online = lambda S: S @ w['W'] + w['b']
    q_target = lambda S: S @ w['W_target'] + w['b_target']

    rng = np.random.default_rng(SEED)
    batch = buf.iloc[rng.choice(len(buf), size=BATCH, replace=False)]   # случайный мини-батч
    S = batch[['s_1', 's_2', 's_3']].to_numpy()
    S2 = batch[['next_1', 'next_2', 'next_3']].to_numpy()
    a, r, done = batch['a'].to_numpy(), batch['r'].to_numpy(), batch['done'].to_numpy()
    rows = np.arange(BATCH)

    q_sa = q_online(S)[rows, a]                                       # Q(s, a; θ)
    y_dqn = r + GAMMA * (1 - done) * q_target(S2).max(axis=1)         # max по целевой сети
    a_star = q_online(S2).argmax(axis=1)                              # Double: выбор онлайн-сетью…
    y_double = r + GAMMA * (1 - done) * q_target(S2)[rows, a_star]    # …оценка целевой
    assert np.all(y_dqn >= y_double - 1e-12)

    loss = huber(y_double - q_sa)
    print(f"средняя переоценка DQN относительно Double: {(y_dqn - y_double).mean():.4f}; "
          f"loss батча = {loss.mean():.4f}")
    return pd.DataFrame({'idx': batch['idx'].to_numpy(), 'a': a, 'q_sa': q_sa, 'y_dqn': y_dqn,
                         'a_star': a_star, 'y_double': y_double, 'huber': loss}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
