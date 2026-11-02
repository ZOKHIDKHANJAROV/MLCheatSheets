#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Метрики сегментации: IoU, Dice, pixel accuracy
"""

import pandas as pd
import numpy as np

def scores(t, p):
    tp = np.sum((t == 1) & (p == 1))
    fp = np.sum((t == 0) & (p == 1))
    fn = np.sum((t == 1) & (p == 0))
    tn = np.sum((t == 0) & (p == 0))
    if tp + fp + fn == 0:             # обе маски пустые — идеальный ответ
        iou = dice = 1.0
    else:
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    return iou, dice, (tp + tn) / len(t)


def main():
    df = pd.read_csv('input.csv')
    rows = []
    for img, g in df.groupby('image_id', sort=False):
        iou, dice, acc = scores(g['true'].to_numpy(), g['pred'].to_numpy())
        assert np.isclose(dice, 2 * iou / (1 + iou))           # Dice и IoU связаны монотонно
        rows.append({'image_id': img, 'iou': iou, 'dice': dice, 'pixel_acc': acc})

    # Сверка со sklearn для непустых случаев
    from sklearn.metrics import f1_score, jaccard_score
    g = df[df['image_id'] == 'img1']
    assert np.isclose(jaccard_score(g['true'], g['pred']), rows[0]['iou'])
    assert np.isclose(f1_score(g['true'], g['pred']), rows[0]['dice'])   # Dice = F1 по пикселям

    result = pd.DataFrame(rows)
    mean = result[['iou', 'dice', 'pixel_acc']].mean()
    result = pd.concat([result, pd.DataFrame([{'image_id': 'mean', **mean.to_dict()}])], ignore_index=True)
    print(result.round(3).to_string(index=False))
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
