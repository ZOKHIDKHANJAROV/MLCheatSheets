# PyTorch Lightning

<!-- время: 12 мин -->

> **О чём лекция.** Какую рутину PyTorch Lightning снимает с вас, как разложить проект по методам `LightningModule`, что делает `Trainer` и как одной строкой включить раннюю остановку, чекпоинты, mixed precision и несколько GPU.

## Зачем это нужно

Цикл обучения на чистом PyTorch выглядит одинаково в каждом проекте: перенос на устройство, `zero_grad`, `backward`, `step`, переключение `train/eval`, валидация, логирование, сохранение лучшей модели. Добавьте обучение на нескольких GPU, mixed precision и накопление градиентов — и служебного кода становится больше, чем кода модели. В нём легко ошибиться: забыть `eval()`, перепутать устройства, неправильно усреднить метрики между GPU.

PyTorch Lightning — надстройка над PyTorch, которая забирает эту инженерную часть. Вы пишете только «науку»: модель, шаг обучения, оптимизатор. Модель при этом остаётся обычным `nn.Module`, а код — обычным PyTorch.

## Интуиция: разделение ролей

Lightning разделяет проект на три части:

- **LightningModule** — *что* обучать: модель, функция потерь, шаги обучения и валидации, оптимизатор;
- **LightningDataModule** (по желанию) — *на чём*: загрузка, разбиение и DataLoader'ы;
- **Trainer** — *как*: число эпох, устройства, точность вычислений, колбэки, логирование.

Один и тот же LightningModule можно запустить на ноутбуке на CPU и на кластере из восьми GPU, поменяв лишь аргументы Trainer.

## Как это работает: LightningModule

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L            # новое имя пакета; старое — import pytorch_lightning as pl

class LitClassifier(L.LightningModule):
    def __init__(self, n_in=20, n_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()          # гиперпараметры попадут в чекпоинт
        self.model = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss                          # backward и step Lightning сделает сам

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=True)
        self.log("val_acc", (logits.argmax(1) == y).float().mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
```

Основные методы:

| метод | назначение |
|---|---|
| `forward` | прямой проход (для инференса) |
| `training_step` | потери на одном батче; вернуть loss |
| `validation_step` / `test_step` | метрики на батче валидации / теста |
| `configure_optimizers` | оптимизатор и, при необходимости, scheduler |
| `predict_step` | предсказания для `trainer.predict` |

Обратите внимание, чего в коде нет: `zero_grad()`, `loss.backward()`, `optimizer.step()`, `.to(device)`, `model.eval()` и `torch.no_grad()` при валидации. Всё это делает Trainer. `self.log` сам усредняет метрики по эпохе и между GPU.

## Trainer

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",               # GPU, если есть, иначе CPU
    devices="auto",
    precision="16-mixed",             # mixed precision: быстрее и меньше памяти на GPU
    gradient_clip_val=1.0,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(monitor="val_loss", save_top_k=1),
    ],
)
trainer.fit(LitClassifier(), train_loader, val_loader)
trainer.test(ckpt_path="best", dataloaders=test_loader)
```

Полезные возможности Trainer:

- `devices=4, strategy="ddp"` — обучение на нескольких GPU (Distributed Data Parallel);
- `precision="16-mixed"` или `"bf16-mixed"` — смешанная точность;
- `accumulate_grad_batches=4` — накопление градиентов: эффективный батч в 4 раза больше без роста памяти;
- `fast_dev_run=True` — прогон одного батча обучения и валидации для проверки, что код не падает;
- `overfit_batches=1` — попытка переобучиться на одном батче: если потери не падают до нуля, в модели или данных баг;
- логгеры TensorBoard, CSV, MLflow, Weights & Biases подключаются аргументом `logger`.

## Загрузка и инференс

```python
model = LitClassifier.load_from_checkpoint("path/to/best.ckpt")
model.eval()
with torch.no_grad():
    preds = model(x_new).argmax(1)
```

Благодаря `save_hyperparameters()` модель восстанавливается без повторной передачи аргументов конструктора.

## Пример на Python

Полный минимальный запуск на синтетических данных — модель из раздела выше плюс данные:

```python
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).long() + (X[:, 2] > 1).long()      # 3 класса
train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=256)

trainer = L.Trainer(max_epochs=20, accelerator="auto", logger=False, enable_checkpointing=False)
trainer.fit(LitClassifier(), train_loader, val_loader)
print(trainer.callback_metrics["val_acc"])
```

## Подводные камни

- **Два имени пакета.** Старое `pytorch_lightning` и новое `lightning` (`import lightning as L`). Не смешивайте импорты из обоих в одном проекте — колбэки и Trainer из разных пакетов несовместимы.
- **Ручной `.to(device)` внутри шагов** не нужен и может мешать при нескольких GPU. Новые тензоры внутри модели создавайте на `self.device`.
- **`self.log` в `training_step`** по умолчанию логирует каждый шаг; для эпохальных значений есть `on_epoch=True`.
- **Метрики, требующие всех предсказаний** (ROC-AUC), неверно считать как среднее по батчам — используйте `torchmetrics`, которые корректно накапливают состояние.
- **Магия скрывает ошибки.** Если что-то непонятно, сначала проверьте модель на чистом PyTorch — Lightning не исправляет ошибки в данных и архитектуре.

## Итоги

- Lightning — структурированный PyTorch: вы пишете модель и шаги, Trainer делает инженерную рутину.
- LightningModule содержит `training_step`, `validation_step`, `configure_optimizers`.
- `backward`, `step`, `zero_grad`, перенос на устройство и режимы train/eval выполняются автоматически.
- Ранняя остановка, чекпоинты, mixed precision, multi-GPU и накопление градиентов включаются аргументами Trainer.
- `fast_dev_run` и `overfit_batches` — быстрые способы отладки.

## Проверьте себя

1. Какие строки цикла обучения PyTorch вам больше не нужно писать в Lightning?
2. Как включить раннюю остановку и сохранение лучшей модели?
3. Зачем запускать `Trainer(overfit_batches=1)`?
