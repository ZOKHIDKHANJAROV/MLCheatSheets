#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Байесовская оптимизация: гауссовский процесс и Expected Improvement
"""

import pandas as pd
import numpy as np

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

N_STEPS, ELL = 8, 0.5


def f(x):
    return np.sin(3 * x) + 0.3 * (x - 1) ** 2


def matern52(a, b):
    r = np.abs(a[:, None] - b[None, :]) / ELL
    return (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)


def gp_by_hand(X, y, grid, alpha=1e-6):
    mean, std = y.mean(), y.std()
    yn = (y - mean) / std
    K = matern52(X, X) + alpha * np.eye(len(X))
    Ks = matern52(X, grid)
    mu = Ks.T @ np.linalg.solve(K, yn)
    var = 1 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean + std * mu, std * np.sqrt(np.maximum(var, 0))


def main():
    start = pd.read_csv('input.csv')
    X, y = start['x'].to_numpy(), start['f'].to_numpy()
    grid = np.linspace(-2, 3, 201)

    rows = []
    for step in range(1, N_STEPS + 1):
        gp = GaussianProcessRegressor(kernel=Matern(length_scale=ELL, nu=2.5), alpha=1e-6,
                                      normalize_y=True, optimizer=None).fit(X[:, None], y)
        mu, sigma = gp.predict(grid[:, None], return_std=True)
        mu_h, sigma_h = gp_by_hand(X, y, grid)
        assert np.allclose(mu, mu_h, atol=1e-6) and np.allclose(sigma, sigma_h, atol=1e-5)

        f_best = y.min()
        ei = np.zeros_like(mu)
        ok = sigma >= 1e-12
        z = (f_best - mu[ok]) / sigma[ok]
        ei[ok] = (f_best - mu[ok]) * norm.cdf(z) + sigma[ok] * norm.pdf(z)

        k = int(np.argmax(ei))
        second = np.partition(ei, -2)[-2]
        assert ei[k] - second > 1e-9          # победитель однозначен
        x_next, f_next = grid[k], f(grid[k])
        X, y = np.append(X, x_next), np.append(y, f_next)
        rows.append({'step': step, 'x_next': x_next, 'ei': ei[k], 'f_next': f_next, 'best_f': y.min()})

    print(f"Найдено: x = {X[y.argmin()]:.3f}, f = {y.min():.4f} за {len(y)} вычислений; "
          f"минимум на сетке: {f(grid).min():.4f}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
