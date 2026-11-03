#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Разложение «как Prophet»: тренд с изломом, Фурье и праздники МНК
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

CHANGEPOINT, TEST_START = '2024-01-01', '2024-11-01'


def fourier(t, period, order, prefix):
    cols = {}
    for k in range(1, order + 1):
        cols[f'{prefix}_sin{k}'] = np.sin(2 * np.pi * k * t / period)
        cols[f'{prefix}_cos{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols)


def holiday_features(ds, holidays):
    out = {}
    for name, grp in holidays.groupby('holiday', sort=False):
        flag = np.zeros(len(ds), dtype=bool)
        for _, h in grp.iterrows():                       # окно каждого вхождения праздника
            lo = h['ds'] + pd.Timedelta(days=int(h['lower_window']))
            hi = h['ds'] + pd.Timedelta(days=int(h['upper_window']))
            flag |= ((ds >= lo) & (ds <= hi)).to_numpy()
        out[f'hol_{name}'] = flag.astype(int)
    return pd.DataFrame(out)


def main():
    df = pd.read_csv('input.csv', parse_dates=['ds'])
    holidays = pd.read_csv('holidays.csv', parse_dates=['ds'])
    t = (df['ds'] - df['ds'].iloc[0]).dt.days.to_numpy(dtype=float)
    t_c = (pd.Timestamp(CHANGEPOINT) - df['ds'].iloc[0]).days

    groups = {
        'trend': pd.DataFrame({'t': t, 't_after_cp': np.maximum(0, t - t_c)}),   # g(t) с изломом
        'weekly': fourier(t, 7, 3, 'w'),                                          # s(t), P = 7
        'yearly': fourier(t, 365.25, 2, 'y'),                                     # s(t), P = 365.25
        'holidays': holiday_features(df['ds'], holidays),                         # h(t)
    }
    X = pd.concat(groups.values(), axis=1)
    train = (df['ds'] < TEST_START).to_numpy()
    model = LinearRegression().fit(X[train], df.loc[train, 'y'])
    coef = pd.Series(model.coef_, index=X.columns)

    comp = pd.DataFrame({'ds': df['ds'].dt.strftime('%Y-%m-%d')})
    for name, g in groups.items():
        comp[name] = g.to_numpy() @ coef[g.columns].to_numpy()
    comp['trend'] += model.intercept_                     # свободный член относим к тренду
    comp['yhat'] = comp[['trend', 'weekly', 'yearly', 'holidays']].sum(axis=1)
    assert np.allclose(comp['yhat'], model.predict(X))

    print(f"наклон до излома {coef['t']:.3f}/день, после {coef['t'] + coef['t_after_cp']:.3f}/день")
    print("эффекты праздников:", coef.filter(like='hol_').round(1).to_dict())
    test = comp[~train].reset_index(drop=True)
    mae = np.mean(np.abs(df.loc[~train, 'y'].to_numpy() - test['yhat'].to_numpy()))
    print(f"MAE на тесте: {mae:.2f} (шум ряда ≈ 3)")
    return test.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
