#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Gridworld: оценка политики и итерация по ценности
"""

import pandas as pd
import numpy as np

MOVES = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
PERP = {'U': 'LR', 'D': 'LR', 'L': 'UD', 'R': 'UD'}
GAMMA, STEP_REWARD, TOL = 0.9, -0.04, 1e-10


def main():
    df = pd.read_csv('input.csv')
    grid = {(r, c): cell for r, c, cell in df[['row', 'col', 'cell']].itertuples(index=False)}
    terminal = {s for s, cell in grid.items() if cell in '+-'}
    states = sorted(s for s, cell in grid.items() if cell not in '#+-')

    def move(s, a):
        nxt = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
        return nxt if grid.get(nxt, '#') != '#' else s           # стена или край — остаёмся

    def reward(s2):
        return {'+': 1.0, '-': -1.0}.get(grid[s2], STEP_REWARD)

    def q_value(V, s, a):
        outcomes = [(0.8, move(s, a))] + [(0.1, move(s, p)) for p in PERP[a]]
        return sum(p * (reward(s2) + GAMMA * V[s2]) for p, s2 in outcomes)

    def iterate(backup):
        V = {s: 0.0 for s in grid if grid[s] != '#'}             # терминальные остаются 0
        while True:
            V_new = dict(V)
            for s in states:
                V_new[s] = backup([q_value(V, s, a) for a in 'URDL'])
            delta = max(abs(V_new[s] - V[s]) for s in states)
            V = V_new
            if delta < TOL:
                return V

    v_random = iterate(lambda qs: np.mean(qs))                   # оценка политики π(a|s) = 1/4
    v_star = iterate(max)                                        # уравнение оптимальности Беллмана
    best = {s: max('URDL', key=lambda a: q_value(v_star, s, a)) for s in states}

    result = pd.DataFrame({'row': [s[0] for s in states], 'col': [s[1] for s in states],
                           'v_random': [v_random[s] for s in states], 'v_star': [v_star[s] for s in states],
                           'best_action': [best[s] for s in states]})
    for r in range(3):
        print(' '.join({'#': '#', '+': '+', '-': '-'}.get(grid[(r, c)], best.get((r, c), '?')) for c in range(4)))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
