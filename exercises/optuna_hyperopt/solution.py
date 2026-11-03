#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Один шаг TPE вручную (идея Optuna и Hyperopt)
"""

import pandas as pd
import numpy as np

from scipy.stats import norm

N_GOOD = 6
BANDWIDTH = {'log_lr': 0.3, 'subsample': 0.05}


def parzen_logpdf(x, centers, h):
    # Смесь гауссиан с центрами в пробах: (1/n)·Σ N(x; xᵢ, h²)
    return np.log(norm.pdf(x[:, None], loc=centers[None, :], scale=h).mean(axis=1))


def true_loss(log_lr, subsample):
    # «Истинный» ландшафт, из которого сгенерирована история (без шума) — только для сравнения
    return 0.20 + 0.04 * (log_lr + 1.2) ** 2 + 0.3 * (subsample - 0.8) ** 2


def main():
    trials = pd.read_csv('input.csv')
    cand = pd.read_csv('candidates.csv')

    good = trials.nsmallest(N_GOOD, 'loss')
    bad = trials.drop(good.index)

    log_l = sum(parzen_logpdf(cand[p].to_numpy(), good[p].to_numpy(), h) for p, h in BANDWIDTH.items())
    log_g = sum(parzen_logpdf(cand[p].to_numpy(), bad[p].to_numpy(), h) for p, h in BANDWIDTH.items())
    log_ratio = log_l - log_g
    rank = (-log_ratio).argsort().argsort() + 1

    result = pd.DataFrame({'candidate': cand['candidate'], 'log_l': log_l, 'log_g': log_g,
                           'log_ratio': log_ratio, 'rank': rank})

    chosen = cand.loc[rank == 1].iloc[0]
    random_pick = cand.iloc[0]
    print(f"TPE выбрал кандидата {int(chosen['candidate'])}: истинная ошибка "
          f"{true_loss(chosen['log_lr'], chosen['subsample']):.4f}; "
          f"лучшая в истории: {trials['loss'].min():.4f}; "
          f"случайный (первый) кандидат: {true_loss(random_pick['log_lr'], random_pick['subsample']):.4f}")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
