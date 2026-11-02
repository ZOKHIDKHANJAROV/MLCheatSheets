#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: VAE: репараметризация, KL и ELBO
"""

import pandas as pd
import numpy as np

def main():
    enc = pd.read_csv('input.csv')
    dec = pd.read_csv('decoder.csv')
    mu = enc[['mu1', 'mu2']].to_numpy()
    logvar = enc[['logvar1', 'logvar2']].to_numpy()
    eps = enc[['eps1', 'eps2']].to_numpy()
    x = dec[[f'x{k}' for k in range(1, 5)]].to_numpy()
    x_hat = dec[[f'xhat{k}' for k in range(1, 5)]].to_numpy()

    # Репараметризация: случайность вынесена в ε, градиент проходит через μ и σ
    sigma = np.exp(0.5 * logvar)
    z = mu + sigma * eps

    # KL(N(μ, σ²) || N(0, 1)) в замкнутой форме, сумма по латентным координатам
    kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=1) + 0.0   # + 0.0: без «-0.0»
    # Численная сверка с общей формулой KL = log(1/σ) + (σ² + μ²)/2 − 1/2
    assert np.allclose(kl, np.sum(-np.log(sigma) + (sigma ** 2 + mu ** 2) / 2 - 0.5, axis=1))
    assert np.isclose(kl[0], 0)

    recon = -np.sum(x * np.log(x_hat) + (1 - x) * np.log(1 - x_hat), axis=1)   # BCE, сумма
    elbo = -(recon + kl)

    print("средний loss (−ELBO):", round(float(-elbo.mean()), 4), " средний KL:", round(float(kl.mean()), 4))
    return pd.DataFrame({'id': enc['id'], 'z1': z[:, 0], 'z2': z[:, 1],
                         'kl': kl, 'recon': recon, 'elbo': elbo}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
