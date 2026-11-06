#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Память агента: буфер с вытеснением и подсчёт попаданий
"""

import pandas as pd
import numpy as np

CAPACITY, HALF_LIFE = 5, 10.0


def victim(policy, buf, t):
    if policy == 'fifo':
        return min(buf, key=lambda k: buf[k]['born'])
    if policy == 'lru':
        return min(buf, key=lambda k: buf[k]['last'])
    if policy == 'importance':
        return min(buf, key=lambda k: (buf[k]['imp'], buf[k]['last']))
    value = lambda k: round(0.6 * buf[k]['imp'] + 0.4 * 0.5 ** ((t - buf[k]['last']) / HALF_LIFE), 9)
    return min(buf, key=lambda k: (value(k), buf[k]['last']))


def simulate(events, policy):
    buf, hits, misses, evictions = {}, 0, 0, 0
    for e in events.itertuples():
        if e.kind == 'write':
            if e.key in buf:
                buf[e.key].update(imp=e.importance, last=e.t)
            else:
                if len(buf) == CAPACITY:
                    del buf[victim(policy, buf, e.t)]
                    evictions += 1
                buf[e.key] = {'born': e.t, 'last': e.t, 'imp': e.importance}
        elif e.key in buf:
            hits += 1
            buf[e.key]['last'] = e.t
        else:
            misses += 1
    return hits, misses, evictions


def main():
    events = pd.read_csv('input.csv').sort_values('t')
    rows = []
    for policy in ['fifo', 'lru', 'importance', 'mixed']:
        hits, misses, evictions = simulate(events, policy)
        rows.append({'policy': policy, 'hits': hits, 'misses': misses,
                     'hit_rate': hits / (hits + misses), 'evictions': evictions})
        print(f"{policy:>10}: попаданий {hits}, промахов {misses}, вытеснений {evictions}")
    result = pd.DataFrame(rows)
    result['hit_rate'] = result['hit_rate'].round(4)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
