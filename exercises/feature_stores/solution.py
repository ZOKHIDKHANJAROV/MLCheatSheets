#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Point-in-time correct join без утечки из будущего
"""

import pandas as pd
import numpy as np

TTL = pd.Timedelta('30D')


def pit_join(labels, feats, col, tolerance=None):
    feats = feats.rename(columns={'ts': 'feature_ts'}).assign(ts=lambda d: d['feature_ts'])
    return pd.merge_asof(labels.sort_values('ts'), feats.sort_values('ts'), on='ts', by='user_id',
                         direction='backward', allow_exact_matches=False, tolerance=tolerance)


def slow_value(hist, user, ts, col, ttl=None):
    h = hist[(hist['user_id'] == user) & (hist['ts'] < ts)].sort_values('ts')
    if h.empty or (ttl is not None and ts - h['ts'].iloc[-1] > ttl):
        return -1
    return int(h[col].iloc[-1])


def main():
    labels = pd.read_csv('input.csv', parse_dates=['ts'])
    orders = pd.read_csv('orders_features.csv', parse_dates=['ts'])
    tickets = pd.read_csv('tickets_features.csv', parse_dates=['ts'])

    o = pit_join(labels, orders, 'orders_30d', tolerance=TTL)
    o['orders_age_days'] = (o['ts'] - o['feature_ts']).dt.days
    t = pit_join(labels, tickets, 'tickets_90d')
    df = (o[['label_id', 'user_id', 'orders_30d', 'orders_age_days']]
          .merge(t[['label_id', 'tickets_90d']], on='label_id')
          .sort_values('label_id').reset_index(drop=True))
    df = df.fillna(-1).astype({'orders_30d': int, 'orders_age_days': int, 'tickets_90d': int})

    # «Наивный» join: последнее значение из всей истории — то, что лежит в онлайн-хранилище сейчас
    latest = orders.sort_values('ts').groupby('user_id')['orders_30d'].last()
    naive = labels.sort_values('label_id')['user_id'].map(latest).to_numpy()
    df['leak'] = (naive != df['orders_30d'].to_numpy()).astype(int)

    # Медленная проверка циклом
    for row, lab in zip(df.itertuples(), labels.sort_values('label_id').itertuples()):
        assert row.orders_30d == slow_value(orders, lab.user_id, lab.ts, 'orders_30d', TTL)
        assert row.tickets_90d == slow_value(tickets, lab.user_id, lab.ts, 'tickets_90d')

    print(f"строк с утечкой при наивном join: {df['leak'].sum()} из {len(df)}")
    return df


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
