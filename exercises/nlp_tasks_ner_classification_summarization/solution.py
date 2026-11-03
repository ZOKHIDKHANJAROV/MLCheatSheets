#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: NER: от BIO-тегов к сущностям и F1 по сущностям
"""

import pandas as pd
import numpy as np

def extract_spans(tags):
    # BIO → [(start, end, type)] по правилам conlleval: «висячий» I-X тоже начинает сущность
    spans, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ['O']):           # 'O' в конце закрывает последнюю сущность
        prefix, _, t = tag.partition('-')
        continues = prefix == 'I' and kind == t
        if start is not None and not continues:
            spans.append((start, i - 1, kind))
            start, kind = None, None
        if prefix == 'B' or (prefix == 'I' and not continues):
            start, kind = i, t
    return spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    df = pd.read_csv('input.csv')
    gold, pred = set(), set()
    for sid, g in df.groupby('sent_id', sort=True):
        gold |= {(sid, *s) for s in extract_spans(g['gold'])}
        pred |= {(sid, *s) for s in extract_spans(g['pred'])}

    rows, totals = [], np.zeros(3, dtype=int)
    for kind in ['LOC', 'ORG', 'PER']:
        g = {s for s in gold if s[3] == kind}
        p = {s for s in pred if s[3] == kind}
        counts = np.array([len(g & p), len(p - g), len(g - p)])
        totals += counts
        rows.append({'entity': kind, 'tp': counts[0], 'fp': counts[1], 'fn': counts[2],
                     **dict(zip(['precision', 'recall', 'f1'], prf(*counts)))})
    rows.append({'entity': 'micro', 'tp': totals[0], 'fp': totals[1], 'fn': totals[2],
                 **dict(zip(['precision', 'recall', 'f1'], prf(*totals)))})

    token_acc = (df['gold'] == df['pred']).mean()
    print(f"точность по токенам = {token_acc:.3f} — выглядит хорошо, но F1 по сущностям = {rows[-1]['f1']:.3f}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
