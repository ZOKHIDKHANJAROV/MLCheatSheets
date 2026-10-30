#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: SGD, Momentum и Adam на «овраге»
"""

import pandas as pd
import numpy as np

f = lambda th: th[0] ** 2 + 10 * th[1] ** 2
grad = lambda th: np.array([2 * th[0], 20 * th[1]])


def run(name, lr, beta1, beta2, eps, steps=50):
    th = np.array([-2.0, 1.5])
    v = m = s = np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(th)
        if name == 'sgd':
            th = th - lr * g
        elif name == 'momentum':
            v = beta1 * v + g
            th = th - lr * v
        elif name == 'adam':
            m = beta1 * m + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g ** 2
            m_hat, s_hat = m / (1 - beta1 ** t), s / (1 - beta2 ** t)
            th = th - lr * m_hat / (np.sqrt(s_hat) + eps)
    return th


def main():
    rows = []
    for _, r in pd.read_csv('input.csv').iterrows():
        th = run(r['optimizer'], r['lr'], r['beta1'], r['beta2'], r['eps'])
        rows.append({'optimizer': r['optimizer'], 'x': th[0], 'y': th[1], 'loss': f(th)})
        print(f"{r['optimizer']:>9}: ({th[0]:+.4f}, {th[1]:+.4f})  f = {f(th):.6f}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
