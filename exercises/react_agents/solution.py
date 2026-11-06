#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Разбор лога ReAct: повторы действий, зацикливание и эффективность
"""

import pandas as pd
import numpy as np

import re
from collections import Counter

ACTION = re.compile(r'^Action: (\w+)\[(.*)\]$')


def norm(text):
    return ' '.join(text.split()).lower()


def analyse(lines):
    steps = format_errors = orphan_obs = 0
    counts = Counter()
    finished = 0
    prev = ''
    for line in lines:
        if line.startswith('Action:'):
            steps += 1
            match = ACTION.match(line)
            if match is None:
                format_errors += 1
            else:
                counts[(norm(match.group(1)), norm(match.group(2)))] += 1
        elif line.startswith('Observation:') and not prev.startswith('Action:'):
            orphan_obs += 1                              # наблюдение без действия — выдумано моделью
        elif line.startswith('Final Answer:'):
            finished = 1
        prev = line
    repeats = sum(c - 1 for c in counts.values())
    return {'steps': steps, 'repeats': repeats, 'looped': int(any(c >= 3 for c in counts.values())),
            'format_errors': format_errors, 'orphan_obs': orphan_obs, 'finished': finished,
            'efficiency': round(len(counts) / steps, 4) if steps else 0.0}


def main():
    log = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
    log['line_no'] = log['line_no'].astype(int)
    log['text'] = log['text'].str.strip()
    log = log.sort_values(['run_id', 'line_no'])
    rows = []
    for run_id, group in log.groupby('run_id'):
        rows.append({'run_id': run_id, **analyse(group['text'].tolist())})
    result = pd.DataFrame(rows).sort_values('run_id').reset_index(drop=True)
    print(f"зациклились: {int(result['looped'].sum())} из {len(result)}")
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
