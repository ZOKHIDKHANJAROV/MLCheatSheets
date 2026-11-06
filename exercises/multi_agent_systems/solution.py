#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Стоимость координации: supervisor, все-со-всеми и конвейер
"""

import pandas as pd
import numpy as np

ROUNDS = 3


def main():
    ag = pd.read_csv('input.csv').sort_values('agent_id')
    ids = [int(a) for a in ag['agent_id']]
    msg = dict(zip(ids, ag['msg_tokens']))
    price = dict(zip(ids, ag['price_per_1k']))
    boss = int(ag.loc[ag['role'] == 'orchestrator', 'agent_id'].iloc[0])
    workers = [a for a in ids if a != boss]

    schemes = {
        'supervisor': [(boss, w) for w in workers] + [(w, boss) for w in workers],
        'all_to_all': [(a, b) for a in ids for b in ids if a != b],
        'pipeline': list(zip(ids, ids[1:])),
    }
    rows = []
    for name, edges in schemes.items():
        inbox = {a: 0 for a in ids}
        tokens, cost = 0, 0.0
        for s, r in edges:
            t = int(msg[s]) * ROUNDS
            tokens += t
            inbox[r] += t
            cost += t * price[r] / 1000
        top = min(ids, key=lambda a: (-inbox[a], a))
        rows.append({'scheme': name, 'messages': len(edges) * ROUNDS, 'tokens': tokens,
                     'cost_usd': round(cost, 4), 'busiest_agent': top, 'busiest_tokens': inbox[top]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
