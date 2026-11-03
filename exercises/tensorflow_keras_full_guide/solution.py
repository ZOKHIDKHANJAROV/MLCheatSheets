#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Колбэки Keras: EarlyStopping и ReduceLROnPlateau по истории обучения
"""

import pandas as pd
import numpy as np

INITIAL_LR, MAX_EPOCHS = 1e-3, 40


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.wait, self.best_epoch = np.inf, 0, None

    def on_epoch_end(self, epoch, val_loss):
        self.wait += 1
        if val_loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = val_loss, epoch, 0
            return False
        return self.wait >= self.patience                 # model.stop_training = True


class ReduceLROnPlateau:
    def __init__(self, factor, patience, min_delta, min_lr):
        self.factor, self.patience, self.min_delta, self.min_lr = factor, patience, min_delta, min_lr
        self.best, self.wait = np.inf, 0

    def on_epoch_end(self, epoch, val_loss, lr):
        if val_loss < self.best - self.min_delta:
            self.best, self.wait = val_loss, 0
        else:
            self.wait += 1
            if self.wait >= self.patience and lr > self.min_lr:
                lr = max(lr * self.factor, self.min_lr)
                self.wait = 0
        return lr


def main():
    history = pd.read_csv('input.csv')
    es = EarlyStopping(patience=5, min_delta=0.0)
    rlr = ReduceLROnPlateau(factor=0.5, patience=2, min_delta=0.005, min_lr=1e-4)

    lr, rows = INITIAL_LR, []
    for epoch, val_loss in zip(history['epoch'], history['val_loss']):
        stop = es.on_epoch_end(epoch, val_loss)
        new_lr = rlr.on_epoch_end(epoch, val_loss, lr)
        rows.append({'epoch': epoch, 'val_loss': val_loss, 'lr': lr, 'es_wait': es.wait,
                     'rlr_wait': rlr.wait, 'best_epoch': es.best_epoch})
        if new_lr != lr:
            print(f"эпоха {epoch}: ReduceLROnPlateau → lr = {new_lr:g}")
        lr = new_lr
        if stop:
            print(f"эпоха {epoch}: EarlyStopping, восстановлены веса эпохи {es.best_epoch}")
            break

    result = pd.DataFrame(rows)
    result['lr'] = result['lr'].round(6)
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
