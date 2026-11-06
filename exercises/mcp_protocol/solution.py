#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Разбор лога JSON-RPC: запросы, ответы, ошибки и таймауты
"""

import pandas as pd
import numpy as np

import json

TIMEOUT_MS = 5000


def main():
    log = pd.read_csv('input.csv')
    pending, rows = {}, []
    for r in log.itertuples():
        msg = json.loads(r.raw)
        if 'method' in msg and 'id' in msg:                       # запрос
            row = {'line_no': int(r.line_no), 'id': int(msg['id']), 'method': msg['method'],
                   'requester': 'client' if r.direction == 'C->S' else 'server',
                   'ts': int(r.ts_ms), 'latency_ms': None, 'error_code': None}
            pending[(r.direction, msg['id'])] = row
            rows.append(row)
        elif 'method' not in msg and 'id' in msg:                 # ответ идёт в обратную сторону
            req_dir = 'S->C' if r.direction == 'C->S' else 'C->S'
            row = pending.get((req_dir, msg['id']))
            if row is None or row['latency_ms'] is not None:      # «сирота» или повторный ответ
                continue
            row['latency_ms'] = int(r.ts_ms) - row['ts']
            if 'error' in msg:
                row['error_code'] = int(msg['error']['code'])
        # уведомления (method без id) ответа не ждут

    for row in rows:
        if row['latency_ms'] is None or row['latency_ms'] > TIMEOUT_MS:
            row['status'] = 'timeout'
        else:
            row['status'] = 'error' if row['error_code'] is not None else 'ok'
    out = pd.DataFrame(rows).drop(columns='ts')[['line_no', 'id', 'method', 'requester', 'status',
                                                 'latency_ms', 'error_code']]
    out['latency_ms'] = out['latency_ms'].astype('Int64')
    out['error_code'] = out['error_code'].astype('Int64')
    print(out['status'].value_counts().to_dict())
    return out


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
