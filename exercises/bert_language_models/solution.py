#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: BERT MLM: маскирование 80/10/10 и потери только на масках
"""

import pandas as pd
import numpy as np

SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}


def main():
    df = pd.read_csv('input.csv')
    special = df['token'].isin(SPECIAL)
    selected = ~special & (df['u_select'] < 0.15)      # служебные токены не маскируем никогда

    sel = df[selected].copy()
    u = sel['u_action']
    sel['action'] = np.select([u < 0.8, u < 0.9], ['mask', 'random'], 'keep')   # 80 / 10 / 10
    sel['input_token'] = np.select([sel['action'] == 'mask', sel['action'] == 'random'],
                                   ['[MASK]', sel['random_token']], sel['token'])
    sel['nll'] = -np.log(sel['p_true'])                # −log p(исходный токен)

    loss = sel['nll'].mean()
    wrong = -np.log(df.loc[~special, 'p_true']).mean()
    print(f"выбрано {len(sel)} из {(~special).sum()} обычных токенов; действия: {sel['action'].value_counts().to_dict()}")
    print(f"MLM loss = {loss:.4f}  (если ошибочно усреднить по всем токенам: {wrong:.4f})")

    result = sel[['seq', 'pos', 'token', 'input_token', 'action', 'nll']].sort_values(['seq', 'pos'])
    total = pd.DataFrame([{'seq': 0, 'pos': 0, 'token': '[ALL]', 'input_token': '[ALL]',
                           'action': 'mean', 'nll': loss}])
    return pd.concat([result, total], ignore_index=True).round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
