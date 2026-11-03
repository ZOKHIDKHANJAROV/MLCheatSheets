#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Трекинг экспериментов: версия данных по хешу и выбор лучшего запуска
"""

import pandas as pd
import numpy as np

import hashlib
import json
from pathlib import Path


def data_version(path):
    text = Path(path).read_text(encoding='utf-8')          # \r\n → \n
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]


def main():
    current = data_version('input.csv')
    runs = [json.loads(line) for line in Path('runs.jsonl').read_text(encoding='utf-8').splitlines() if line]
    df = pd.json_normalize(runs)
    print('текущая версия данных:', current)

    df['data_ok'] = (df['data_version'] == current).astype(int)
    df['eligible'] = ((df['data_ok'] == 1) & (df['status'] == 'FINISHED')).astype(int)

    ok = df[df['eligible'] == 1].sort_values(
        ['metrics.cv_auc', 'metrics.cv_auc_std', 'run_id'], ascending=[False, True, True])
    df['rank'] = 0
    df.loc[ok.index, 'rank'] = np.arange(1, len(ok) + 1)
    df['best'] = (df['rank'] == 1).astype(int)

    best = df[df['best'] == 1].iloc[0]
    print('лучший запуск:', best['run_id'], best.filter(like='params.').dropna().to_dict(),
          'cv_auc =', best['metrics.cv_auc'])
    trap = df.loc[df['metrics.cv_auc'].idxmax(), 'run_id']
    print('максимальный cv_auc во всём журнале у', trap, '— но он не допустим')
    return df[['run_id', 'data_ok', 'eligible', 'rank', 'best']]


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
