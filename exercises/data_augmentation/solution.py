#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Геометрические аугментации: картинка и рамка синхронно
"""

import pandas as pd
import numpy as np

def apply(op, img, box, r):
    H, W = img.shape
    x0, y0, x1, y1 = box
    if op == 'hflip':
        return img[:, ::-1], (W - x1, y0, W - x0, y1)          # min/max меняются местами
    if op == 'vflip':
        return img[::-1], (x0, H - y1, x1, H - y0)
    if op == 'rot90':
        # (x, y) → (y, W − x): x-диапазон берётся из y, y-диапазон — из отражённого x
        return np.rot90(img), (y0, W - x1, y1, W - x0)
    if op == 'crop':
        t, l, h, w = int(r.top), int(r.left), int(r.height), int(r.width)
        clip = lambda v, hi: min(max(v, 0), hi)
        return img[t:t + h, l:l + w], (clip(x0 - l, w), clip(y0 - t, h), clip(x1 - l, w), clip(y1 - t, h))
    raise ValueError(op)


def main():
    px = pd.read_csv('input.csv')
    img = px.pivot(index='row', columns='col', values='value').to_numpy()
    box = tuple(pd.read_csv('box.csv').iloc[0])
    ops = pd.read_csv('ops.csv')

    def record(step, op):
        x0, y0, x1, y1 = box
        return {'step': step, 'op': op, 'height': img.shape[0], 'width': img.shape[1],
                'x_min': x0, 'y_min': y0, 'x_max': x1, 'y_max': y1,
                'box_area': (x1 - x0) * (y1 - y0), 'box_sum': int(img[y0:y1, x0:x1].sum()),
                'corner': int(img[0, 0])}

    rows = [record(0, 'original')]
    for r in ops.itertuples():
        img, box = apply(r.op, img, box, r)
        rows.append(record(r.step, r.op))
        if r.op != 'crop':                              # геометрия без потерь: содержимое рамки то же
            assert rows[-1]['box_sum'] == rows[0]['box_sum']

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
