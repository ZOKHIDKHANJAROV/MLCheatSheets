#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Q-learning и SARSA по записанной траектории
"""

import pandas as pd
import numpy as np

ALPHA, GAMMA = 0.5, 0.9


def main():
    df = pd.read_csv('input.csv')
    q = np.zeros((5, 2))
    sarsa = np.zeros((5, 2))
    for _, r in df.iterrows():
        s, a, rew, s2, a2, done = int(r['s']), int(r['a']), r['r'], int(r['s_next']), int(r['a_next']), int(r['done'])
        q_target = rew if done else rew + GAMMA * q[s2].max()               # off-policy: лучшее действие
        q[s, a] += ALPHA * (q_target - q[s, a])
        s_target = rew if done else rew + GAMMA * sarsa[s2, a2]             # on-policy: реально выбранное
        sarsa[s, a] += ALPHA * (s_target - sarsa[s, a])
    print("Q-learning:\n", q[:4].round(3), "\nSARSA:\n", sarsa[:4].round(3))
    return pd.DataFrame({'s': range(4), 'q_left': q[:4, 0], 'q_right': q[:4, 1],
                         'sarsa_left': sarsa[:4, 0], 'sarsa_right': sarsa[:4, 1]}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
