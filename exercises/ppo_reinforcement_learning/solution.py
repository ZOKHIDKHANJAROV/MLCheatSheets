#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: PPO: преимущества GAE и clipped-цель по шагам роллаута
"""

import pandas as pd
import numpy as np

GAMMA, LAM, EPS = 0.99, 0.95, 0.2


def main():
    df = pd.read_csv('input.csv').sort_values('t')
    r, done = df['reward'].to_numpy(), df['done'].to_numpy()
    v, nv = df['value'].to_numpy(), df['next_value'].to_numpy()

    delta, adv, last = np.zeros(len(df)), np.zeros(len(df)), 0.0
    for t in reversed(range(len(df))):
        nonterminal = 1.0 - done[t]
        delta[t] = r[t] + GAMMA * nonterminal * nv[t] - v[t]       # TD-ошибка критика
        last = delta[t] + GAMMA * LAM * nonterminal * last         # GAE не переходит границу эпизода
        adv[t] = last
    ret = adv + v                                                  # цель для критика

    adv_norm = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)
    ratio = np.exp(df['logp_new'].to_numpy() - df['logp_old'].to_numpy())
    l_clip = np.minimum(ratio * adv_norm, np.clip(ratio, 1 - EPS, 1 + EPS) * adv_norm)
    clipped = ((adv_norm > 0) & (ratio > 1 + EPS)) | ((adv_norm < 0) & (ratio < 1 - EPS))

    print(f"L_CLIP (среднее) = {l_clip.mean():.4f}, без обрезки = {(ratio * adv_norm).mean():.4f}, "
          f"clip fraction = {clipped.mean():.2f}")
    return pd.DataFrame({'t': df['t'], 'delta': delta, 'adv': adv, 'ret': ret, 'adv_norm': adv_norm,
                         'ratio': ratio, 'l_clip': l_clip, 'clipped': clipped.astype(int)}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
