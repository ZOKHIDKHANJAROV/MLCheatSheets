#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Симулятор цикла агента: шаги, стоимость и успешность при лимите шагов
"""

import pandas as pd
import numpy as np

MAX_STEPS = 6
PRICE_PER_1K = 0.5


def task_status(group):
    finish = group[group['action'] == 'finish']
    if finish.empty:
        return 'limit'                                # ответа в пределах лимита нет
    return 'success' if finish['correct'].iloc[0] == 1 else 'wrong'


def main():
    log = pd.read_csv('input.csv')
    log = log[log['step'] <= MAX_STEPS]               # шаги после лимита не существуют
    rows = []
    for task_id, group in log.groupby('task_id'):
        tokens = int(group['tokens'].sum())
        rows.append({'task_id': task_id, 'steps_used': len(group), 'tokens': tokens,
                     'cost': round(tokens / 1000 * PRICE_PER_1K, 4), 'status': task_status(group)})
    result = pd.DataFrame(rows).sort_values('task_id').reset_index(drop=True)
    print(result['status'].value_counts().to_string())
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
