#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: ViT: нарезка на патчи, линейная проекция, [CLS] и позиции
"""

import pandas as pd
import numpy as np

import json

P = 3


def main():
    df = pd.read_csv('input.csv').sort_values(['row', 'col', 'channel'])
    img = df['value'].to_numpy().reshape(6, 6, 2)             # (H, W, C)
    w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
    H, W, C = img.shape
    gh, gw = H // P, W // P

    patches = img.reshape(gh, P, gw, P, C).transpose(0, 2, 1, 3, 4).reshape(gh * gw, P * P * C)
    assert np.allclose(patches[1], img[0:3, 3:6, :].reshape(-1))   # патч (0, 1) — правый верхний

    tokens = np.vstack([w['cls'], patches @ w['E'] + w['b']]) + w['pos']   # [CLS] + патчи, + позиции
    print(f"патчей: {len(patches)}, токенов: {len(tokens)}, размерность: {tokens.shape[1]}")

    result = pd.DataFrame(tokens.round(4), columns=['z1', 'z2', 'z3', 'z4'])
    result.insert(0, 'token', np.arange(len(tokens)))
    result.insert(1, 'patch_row', [-1] + [k // gw for k in range(gh * gw)])
    result.insert(2, 'patch_col', [-1] + [k % gw for k in range(gh * gw)])
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
