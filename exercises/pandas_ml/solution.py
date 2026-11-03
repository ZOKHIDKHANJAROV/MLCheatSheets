#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Pandas: чистка, пропуски по группам, groupby и merge
"""

import pandas as pd
import numpy as np

REPORT_DATE = pd.Timestamp('2026-04-01')


def main():
    customers = pd.read_csv('input.csv')
    orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
    print(orders.dtypes['amount'], '| дублей:', orders.duplicated().sum())

    # 1) Дубли выгрузки
    orders = orders.drop_duplicates()

    # 2) Сумма: десятичная запятая → точка, мусор → NaN → удаляем
    orders['amount'] = pd.to_numeric(orders['amount'].str.replace(',', '.', regex=False), errors='coerce')
    print('нечисловых сумм:', orders['amount'].isna().sum())
    orders = orders.dropna(subset=['amount'])

    # 3) Пропуски возраста — медианой своего города
    city_median = customers.groupby('city')['age'].transform('median')
    customers['age'] = customers['age'].fillna(city_median)

    # 4) Агрегаты по клиенту
    agg = orders.groupby('customer_id').agg(
        n_orders=('order_id', 'size'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        last_order=('order_date', 'max'),
    ).reset_index()
    agg['days_since_last'] = (REPORT_DATE - agg['last_order']).dt.days

    # 5) Left join: клиенты без заказов остаются
    result = customers.merge(agg.drop(columns='last_order'), on='customer_id', how='left')
    assert len(result) == len(customers)
    result = result.fillna({'n_orders': 0, 'total_amount': 0, 'avg_amount': 0, 'days_since_last': -1})
    result[['n_orders', 'days_since_last']] = result[['n_orders', 'days_since_last']].astype(int)

    print(result.groupby('city')[['total_amount']].mean().round(1))
    result = result.sort_values('customer_id').reset_index(drop=True)
    return result.round({'age': 2, 'total_amount': 2, 'avg_amount': 2})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
