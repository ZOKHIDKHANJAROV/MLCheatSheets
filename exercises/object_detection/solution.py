#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: IoU и Non-Maximum Suppression
"""

import pandas as pd
import numpy as np

IOU_THR = 0.5


def iou(a, b):
    # a, b — (x1, y1, x2, y2)
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def main():
    df = pd.read_csv('input.csv')
    boxes = {r.box_id: (r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()}

    # В данных нет IoU, «висящих» на пороге: ответ не зависит от > или >=
    ious = [iou(boxes[a], boxes[b]) for a in boxes for b in boxes if a < b]
    assert all(abs(v - IOU_THR) > 1e-6 for v in ious)

    kept = []
    for cls, g in df.groupby('cls'):
        order = g.sort_values(['score', 'box_id'], ascending=[False, True])['box_id'].tolist()
        while order:
            best = order.pop(0)
            rest = [b for b in order if iou(boxes[best], boxes[b]) <= IOU_THR]
            kept.append({'box_id': best, 'n_suppressed': len(order) - len(rest)})
            order = rest

    result = pd.DataFrame(kept).merge(df[['box_id', 'cls', 'score']], on='box_id')
    result = result.sort_values(['score', 'box_id'], ascending=[False, True])
    print(f"оставлено {len(result)} из {len(df)} рамок")
    return result[['box_id', 'cls', 'score', 'n_suppressed']].reset_index(drop=True)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
