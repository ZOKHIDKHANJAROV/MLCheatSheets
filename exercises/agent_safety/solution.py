#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Безопасность агента: политика вызовов инструментов и «смертельная триада»
"""

import pandas as pd
import numpy as np

KNOWN = {'read_file', 'write_file', 'query_db', 'read_email', 'http_get', 'http_post', 'send_email',
         'run_shell', 'delete_db', 'transfer_money'}
CAPABILITY = {'read_file': 'private_data', 'query_db': 'private_data',
              'http_get': 'untrusted_content', 'read_email': 'untrusted_content',
              'http_post': 'external_channel', 'send_email': 'external_channel'}
CAPS = ['private_data', 'untrusted_content', 'external_channel']


def decide(tool, target, amount):
    if tool not in KNOWN:
        return 'deny'                                  # deny by default
    if '..' in target:
        return 'deny'                                  # выход из каталога
    if tool in ('run_shell', 'delete_db'):
        return 'deny'
    if tool == 'read_file':
        if target.startswith('/etc/') or '.ssh' in target or target.endswith('.env'):
            return 'deny'
        return 'allow' if target.startswith('/work/') else 'ask'
    if tool == 'write_file':
        if target.startswith('/work/tmp/'):
            return 'allow'
        return 'ask' if target.startswith('/work/') else 'deny'
    if tool == 'query_db':
        return 'ask' if target in ('users_pii', 'payments') else 'allow'
    if tool == 'read_email':
        return 'allow'
    if tool == 'http_get':
        return 'allow' if target in ('docs.python.org', 'api.company.com') else 'ask'
    if tool == 'http_post':
        return 'ask' if target == 'api.company.com' else 'deny'
    if tool == 'send_email':
        return 'ask' if target.endswith('@company.com') else 'deny'
    return 'ask' if float(amount) <= 100 else 'deny'   # transfer_money


def main():
    df = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
    df['verdict'] = [decide(t, g, a) for t, g, a in zip(df['tool'], df['target'], df['amount'])]
    df['session_id'] = df['session_id'].astype(int)

    rows = []
    for sid, g in df.groupby('session_id', sort=True):
        alive = g[g['verdict'] != 'deny']              # запрещённые вызовы возможностей не дают
        have = set(alive['tool'].map(CAPABILITY).dropna())
        row = {'session_id': sid}
        row.update({v: int((g['verdict'] == v).sum()) for v in ('allow', 'ask', 'deny')})
        row.update({cap: int(cap in have) for cap in CAPS})
        row['trifecta'] = int(all(cap in have for cap in CAPS))
        rows.append(row)
        if row['trifecta']:
            print(f"сессия {sid}: триада! уберите одну из трёх возможностей")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
