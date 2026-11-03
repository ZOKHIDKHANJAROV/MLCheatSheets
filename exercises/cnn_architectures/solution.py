#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Архитектура CNN: формы выходов и число параметров по слоям
"""

import pandas as pd
import numpy as np

def main():
    layers = pd.read_csv('input.csv')
    C, H, W = 3, 32, 32
    rows, total = [], 0
    for L in layers.itertuples():
        if L.type == 'conv':
            k, s, p = int(L.kernel), int(L.stride), int(L.padding)
            params = k * k * C * int(L.out) + (int(L.out) if L.bias == 1 else 0)
            C, H, W = int(L.out), (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
        elif L.type == 'maxpool':
            k, s, p = int(L.kernel), int(L.stride), int(L.padding)
            params = 0
            H, W = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
        elif L.type == 'bn':
            params = 2 * C                          # γ и β на каждый канал
        elif L.type == 'flatten':
            params = 0
            C, H, W = C * H * W, 1, 1
        elif L.type == 'dense':
            params = C * int(L.out) + int(L.out)
            C, H, W = int(L.out), 1, 1
        else:                                       # relu
            params = 0
        total += params
        rows.append({'layer_id': L.layer_id, 'type': L.type, 'out_c': C, 'out_h': H, 'out_w': W,
                     'params': params, 'cum_params': total})

    result = pd.DataFrame(rows)
    conv_share = result.loc[result['type'] == 'conv', 'params'].sum() / total
    print(f"всего параметров: {total}, из них в свёртках: {conv_share:.0%}")
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
