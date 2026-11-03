#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Диффузия: прямой процесс q(x_t | x₀) и шаг обратного DDPM
"""

import pandas as pd
import numpy as np

T = 1000


def main():
    df = pd.read_csv('input.csv')
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    t = df['t'].to_numpy()
    b, a, ab = betas[t - 1], alphas[t - 1], alpha_bar[t - 1]   # шаги нумеруются с 1
    x0, eps, eps_pred = df['x0'], df['eps'], df['eps_pred']

    x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps               # q(x_t | x_0) одним прыжком
    x0_hat = (x_t - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
    mu = (x_t - b / np.sqrt(1 - ab) * eps_pred) / np.sqrt(a)     # среднее p_θ(x_{t−1} | x_t)

    # Проверка: при идеальной сети μ совпадает со средним апостериора q(x_{t−1} | x_t, x_0)
    ab_prev = np.where(t > 1, alpha_bar[np.maximum(t - 2, 0)], 1.0)
    mu_ideal = (x_t - b / np.sqrt(1 - ab) * eps) / np.sqrt(a)
    mu_post = np.sqrt(ab_prev) * b / (1 - ab) * x0 + np.sqrt(a) * (1 - ab_prev) / (1 - ab) * x_t
    assert np.allclose(mu_ideal, mu_post)
    print(pd.DataFrame({'t': t, 'signal': np.sqrt(ab), 'noise': np.sqrt(1 - ab)}).drop_duplicates().round(3))

    return pd.DataFrame({'t': t, 'k': df['k'], 'alpha_bar': ab.round(6), 'x_t': x_t.round(4),
                         'x0_hat': x0_hat.round(4), 'mu': mu.round(4)})


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
