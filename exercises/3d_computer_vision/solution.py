#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Стереозрение: глубина по диспаритету и 3D-точки
"""

import pandas as pd
import numpy as np

import json


def main():
    df = pd.read_csv('input.csv')
    cam = json.load(open('camera.json', encoding='utf-8'))
    f, cx, cy, B = cam['f'], cam['cx'], cam['cy'], cam['baseline_m']

    d = df['u_left'] - df['u_right']          # диспаритет: насколько точка «сдвинулась» между кадрами
    Z = f * B / d                              # глубина из подобия треугольников
    X = (df['u_left'] - cx) * Z / f            # обращение u = f·X/Z + cx
    Y = (df['v'] - cy) * Z / f
    dZ = Z ** 2 / (f * B)                      # ошибка глубины при ошибке d на 1 пиксель

    # Обратная проекция в обе камеры (правая сдвинута на B по оси X)
    assert np.allclose(f * X / Z + cx, df['u_left'])
    assert np.allclose(f * (X - B) / Z + cx, df['u_right'])

    result = pd.DataFrame({'point_id': df['point_id'], 'disparity': d, 'Z': Z, 'X': X, 'Y': Y,
                           'dZ_1px': dZ})
    far = result.loc[result['Z'].idxmax()]
    print(f"самая далёкая точка: Z = {far['Z']:.1f} м, ошибка на 1 px ≈ {far['dZ_1px']:.1f} м")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
