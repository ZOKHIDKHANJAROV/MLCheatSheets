# PyTorch Lightning — вопросы для самопроверки

### Какую проблему решает PyTorch Lightning?
Убирает шаблонный код обучения: цикл по эпохам, перенос на устройства, логирование, чекпоинты, распределённое обучение. Исследовательская логика остаётся в `LightningModule`, «инженерия» — в `Trainer`.

### Какие методы обычно определяют в `LightningModule`?
`forward` (по желанию), `training_step` (возвращает loss), `validation_step`, `configure_optimizers`; при необходимости `test_step` и `predict_step`.

### Нужно ли в Lightning вызывать `loss.backward()`, `optimizer.step()` и `.to(device)` вручную?
Нет: при автоматической оптимизации `Trainer` делает это сам. Достаточно вернуть loss из `training_step`, а устройство задаётся параметрами `Trainer(accelerator=..., devices=...)`.

### Как в Lightning включить раннюю остановку и сохранение лучшей модели?
Передать в `Trainer(callbacks=[...])` колбэки `EarlyStopping(monitor='val_loss')` и `ModelCheckpoint(monitor='val_loss')`; метрику нужно логировать через `self.log('val_loss', loss)`.

### Как в Lightning обучать в смешанной точности и на нескольких GPU?
Параметрами `Trainer`: `precision='16-mixed'` (или `'bf16-mixed'`), `devices=4`, `strategy='ddp'`. Код модели при этом менять не нужно.
