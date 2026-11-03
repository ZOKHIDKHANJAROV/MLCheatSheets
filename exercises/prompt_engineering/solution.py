#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Оценка вариантов промпта: exact match и self-consistency
"""

import pandas as pd
import numpy as np

import re
from collections import Counter


def normalize(text):
    text = str(text)
    if 'Ответ:' in text:
        text = text.rsplit('Ответ:', 1)[1]                 # часть после последнего «Ответ:»
    return ' '.join(text.split()).lower().rstrip('.').strip()


def majority(answers):
    counts = Counter(answers)
    best = max(counts.values())
    return next(a for a in answers if counts[a] == best)  # тай-брейк: кто встретился раньше


def main():
    outs = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
    gold = pd.read_csv('questions.csv', dtype=str, keep_default_na=False)
    gold = dict(zip(gold['question_id'], gold['answer'].map(normalize)))
    outs['norm'] = outs['output'].map(normalize)
    outs['sample'] = outs['sample'].astype(int)

    rows = []
    for variant in ['zero_shot', 'few_shot', 'cot']:
        part = outs[outs['variant'] == variant]
        first, vote, agree = [], [], []
        for qid, g in part.groupby('question_id'):
            answers = g.sort_values('sample')['norm'].tolist()
            maj = majority(answers)
            first.append(answers[0] == gold[qid])
            vote.append(maj == gold[qid])
            agree.append(answers.count(maj) / len(answers))
        fmt = part['norm'].map(lambda s: re.fullmatch(r'-?\d+', s) is not None).mean()
        rows.append({'variant': variant, 'em_first': np.mean(first), 'em_majority': np.mean(vote),
                     'agreement': np.mean(agree), 'format_ok': fmt})

    result = pd.DataFrame(rows).round(4)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
