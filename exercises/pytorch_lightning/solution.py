#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Trainer Lightning на NumPy: логирование, EarlyStopping и ModelCheckpoint
"""

import pandas as pd
import numpy as np

MAX_EPOCHS = 15


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.wait = np.inf, 0

    def on_validation_end(self, epoch, metrics):
        current = metrics['val_loss']
        if current < self.best - self.min_delta:            # улучшение больше min_delta
            self.best, self.wait = current, 0
        else:
            self.wait += 1
        return self.wait >= self.patience                   # trainer.should_stop


class ModelCheckpoint:
    def __init__(self, save_top_k):
        self.k, self.saved = save_top_k, {}                 # эпоха → val_loss

    def on_validation_end(self, epoch, metrics):
        score = metrics['val_loss']
        if len(self.saved) < self.k or score < max(self.saved.values()):
            if len(self.saved) == self.k:
                del self.saved[max(self.saved, key=self.saved.get)]   # удаляем худший
            self.saved[epoch] = score
        return False

    @property
    def best_epoch(self):
        return min(self.saved, key=self.saved.get)


def weighted(g, col):
    return np.average(g[col], weights=g['batch_size'])       # как self.log(on_epoch=True, batch_size=…)


def main():
    logs = pd.read_csv('input.csv')
    logs = logs[logs['stage'] != 'sanity_check']            # sanity check не логируется
    ckpt = ModelCheckpoint(save_top_k=2)
    es = EarlyStopping(patience=3, min_delta=0.01)

    rows = []
    for epoch in range(MAX_EPOCHS):
        train = logs[(logs['stage'] == 'train') & (logs['epoch'] == epoch)]
        val = logs[(logs['stage'] == 'val') & (logs['epoch'] == epoch)]
        metrics = {'train_loss': weighted(train, 'loss'), 'val_loss': weighted(val, 'loss'),
                   'val_acc': weighted(val, 'acc')}
        ckpt.on_validation_end(epoch, metrics)
        should_stop = es.on_validation_end(epoch, metrics)
        rows.append({'epoch': epoch, **metrics, 'es_wait': es.wait})
        if should_stop:
            print(f"EarlyStopping: остановка после эпохи {epoch}")
            break

    result = pd.DataFrame(rows)
    result['in_top_k'] = result['epoch'].isin(list(ckpt.saved)).astype(int)
    result['is_best'] = (result['epoch'] == ckpt.best_epoch).astype(int)
    print(f"сохранены эпохи {sorted(ckpt.saved)}, best_model_path → epoch={ckpt.best_epoch}")
    return result.round(4)


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
