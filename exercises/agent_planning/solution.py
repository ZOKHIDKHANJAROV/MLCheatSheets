#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: План агента: порядок подзадач, критический путь и бюджет токенов
"""

import pandas as pd
import numpy as np

import heapq
import json


def main():
    df = pd.read_csv('input.csv', dtype={'deps': str}, keep_default_na=False)
    budget = json.load(open('budget.json', encoding='utf-8'))['token_budget']

    deps = {int(r.task_id): [int(x) for x in r.deps.split(';') if x] for r in df.itertuples()}
    dur = dict(zip(df['task_id'], df['duration']))
    tok = dict(zip(df['task_id'], df['tokens']))
    children = {t: [] for t in deps}
    for t, ps in deps.items():
        for p in ps:
            children[p].append(t)

    # 1. алгоритм Кана: из готовых берём наименьший task_id
    indeg = {t: len(ps) for t, ps in deps.items()}
    ready = [t for t, k in indeg.items() if k == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        t = heapq.heappop(ready)
        order.append(t)
        for c in children[t]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, c)
    assert len(order) == len(deps), 'в графе есть цикл'

    # 2. прямой проход: самое раннее начало и окончание
    start, finish = {}, {}
    for t in order:
        start[t] = max((finish[p] for p in deps[t]), default=0)
        finish[t] = start[t] + dur[t]
    total = max(finish.values())

    # 3. обратный проход: самое позднее окончание и запас
    lf = {}
    for t in reversed(order):
        lf[t] = min((lf[c] - dur[c] for c in children[t]), default=total)

    # 4. бюджет токенов
    done, spent, rows = set(), 0, []
    for i, t in enumerate(order, start=1):
        if any(p not in done for p in deps[t]):
            status = 'blocked'
        elif spent + tok[t] > budget:
            status = 'skipped_budget'
        else:
            status = 'done'
            done.add(t)
            spent += int(tok[t])
        rows.append({'order': i, 'task_id': t, 'start': int(start[t]), 'finish': int(finish[t]),
                     'slack': int(lf[t] - finish[t]), 'status': status, 'spent': spent})
    print('длительность плана:', total, '| потрачено токенов:', spent, 'из', budget)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
