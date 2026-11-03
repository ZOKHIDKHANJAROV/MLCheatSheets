#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: REINFORCE: возвраты, baseline и градиент softmax-политики
"""

import pandas as pd
import numpy as np

GAMMA, LR = 0.9, 0.5


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    episodes = pd.read_csv('input.csv').sort_values(['episode', 't'])
    theta = pd.read_csv('theta.csv').sort_values('s')[['a0', 'a1', 'a2']].to_numpy()
    pi = softmax(theta)

    # Возвраты reward-to-go: с конца эпизода
    G = np.zeros(len(episodes))
    for _, idx in episodes.groupby('episode').indices.items():
        g = 0.0
        for i in idx[::-1]:
            g = episodes['r'].iloc[i] + GAMMA * g
            G[i] = g
    baseline = G.mean()
    n_episodes = episodes['episode'].nunique()

    grad, grad_b = np.zeros_like(theta), np.zeros_like(theta)
    for (s, a), g_t in zip(episodes[['s', 'a']].to_numpy(), G):
        glog = -pi[s].copy()
        glog[a] += 1                                 # ∇θ[s] log π(a|s) = one_hot(a) − π(·|s)
        grad[s] += glog * g_t
        grad_b[s] += glog * (g_t - baseline)
    grad /= n_episodes
    grad_b /= n_episodes
    assert np.allclose(grad_b.sum(axis=1), 0)

    theta_new = theta + LR * grad_b                  # градиентный ПОДЪЁМ
    pi_new = softmax(theta_new)
    print(f"baseline = {baseline:.3f}; π до:\n{pi.round(3)}\nπ после:\n{pi_new.round(3)}")

    s_idx, a_idx = np.divmod(np.arange(9), 3)
    return pd.DataFrame({'s': s_idx, 'a': a_idx, 'pi': pi.ravel(), 'grad_reinforce': grad.ravel(),
                         'grad_baseline': grad_b.ravel(), 'theta_new': theta_new.ravel(),
                         'pi_new': pi_new.ravel()}).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
