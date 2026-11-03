#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: YOLO: декодирование сетки и якорей, порог уверенности и NMS
"""

import pandas as pd
import numpy as np

import json


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)


def main():
    df = pd.read_csv('input.csv')
    cfg = json.load(open('config.json', encoding='utf-8'))
    s, size = cfg['stride'], cfg['image_size']
    anchors = np.array(cfg['anchors'], dtype=float)

    bx = (sigmoid(df['tx']) + df['gx']) * s                       # центр в пикселях
    by = (sigmoid(df['ty']) + df['gy']) * s
    bw = anchors[df['anchor'], 0] * np.exp(df['tw'])               # поправка к якорю
    bh = anchors[df['anchor'], 1] * np.exp(df['th'])
    logits = df[['cls_person', 'cls_car']].to_numpy()
    cls_idx = logits.argmax(axis=1)
    score = sigmoid(df['obj']) * sigmoid(logits.max(axis=1))

    det = df[['gy', 'gx', 'anchor']].copy()
    det['cls'] = np.array(cfg['classes'])[cls_idx]
    det['score'] = score
    det['x1'], det['y1'] = np.clip(bx - bw / 2, 0, size), np.clip(by - bh / 2, 0, size)
    det['x2'], det['y2'] = np.clip(bx + bw / 2, 0, size), np.clip(by + bh / 2, 0, size)

    assert np.min(np.abs(det['score'] - cfg['score_threshold'])) > 1e-3   # нет рамок на пороге
    det = det[det['score'] > cfg['score_threshold']]
    print(f"после порога: {len(det)} из {len(df)}")

    keep = []
    for _, g in det.groupby('cls'):
        order = list(g.sort_values('score', ascending=False).index)
        while order:
            best = order.pop(0)
            keep.append(best)
            a = det.loc[best, ['x1', 'y1', 'x2', 'y2']].to_numpy()
            ious = {i: iou(a, det.loc[i, ['x1', 'y1', 'x2', 'y2']].to_numpy()) for i in order}
            assert all(abs(v - cfg['iou_threshold']) > 1e-3 for v in ious.values())
            order = [i for i in order if ious[i] <= cfg['iou_threshold']]

    result = det.loc[keep].sort_values('score', ascending=False).reset_index(drop=True)
    result['score'] = result['score'].round(4)
    result[['x1', 'y1', 'x2', 'y2']] = result[['x1', 'y1', 'x2', 'y2']].round(2)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
