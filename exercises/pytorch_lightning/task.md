# Упражнение: Trainer Lightning на NumPy: логирование, EarlyStopping и ModelCheckpoint

## Задание

Повторите логику `lightning.Trainer` без PyTorch. В `input.csv` — значения, которые
`training_step` и `validation_step` вернули бы на каждом батче, если бы обучение шло все 15 эпох:
`stage` (`sanity_check` / `train` / `val`), `epoch` (с 0, как в Lightning), `batch_idx`, `batch_size`,
`loss`, `acc` (только для валидации).

```python
# в LightningModule: self.log('train_loss', loss, on_epoch=True, batch_size=len(x))
#                    self.log('val_loss', loss, batch_size=len(x)); self.log('val_acc', acc, batch_size=len(x))
trainer = L.Trainer(max_epochs=15, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=0.01),
    ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=2)])
```

Семантика, которую нужно воспроизвести:

1. **Sanity check** — два батча валидации перед обучением; метрики и колбэки их **не видят**.
2. **Метрика эпохи** = среднее по батчам, **взвешенное по `batch_size`** (так `self.log` агрегирует при
   `on_epoch=True`; последний батч обучения меньше остальных).
3. **ModelCheckpoint** после валидации каждой эпохи: если сохранено меньше 2 чекпоинтов или val_loss строго
   меньше худшего из сохранённых — сохранить эпоху (худший при переполнении удаляется). Лучший чекпоинт — с
   минимальным val_loss.
4. **EarlyStopping**: улучшение, если val_loss < best − min_delta; тогда best = val_loss и wait = 0, иначе
   wait += 1 (best при этом не меняется). Если wait ≥ patience — обучение останавливается после этой эпохи.

## Подсказки

- Невзвешенное среднее двух батчей валидации отличается от взвешенного — проверьте на эпохе 0.
- Улучшение меньше min_delta для EarlyStopping не считается, а для ModelCheckpoint — считается: «лучший
  чекпоинт» и «best» колбэка остановки могут разойтись.
- `np.average(g['loss'], weights=g['batch_size'])` по группам `epoch`; колбэки удобно оформить классами
  с методом `on_validation_end(epoch, metrics)` — как в Lightning.

## Ожидаемый формат выходного файла

CSV с колонками `epoch, train_loss, val_loss, val_acc, es_wait, in_top_k, is_best` — по строке на каждую
**пройденную** эпоху (с 0 до эпохи остановки включительно). `es_wait` — счётчик EarlyStopping после этой эпохи;
`in_top_k` = 1, если чекпоинт этой эпохи остался среди сохранённых в конце обучения, иначе 0; `is_best` = 1
только у эпохи лучшего чекпоинта (`best_model_path`). Числа округлить до 4 знаков.

## Проверка

Сохраните свой результат в `my_answer.csv` (в папке упражнения) и выполните из корня репозитория:

```bash
python tools/check_exercise.py exercises/pytorch_lightning
```

Эталонное решение с пояснениями — `solution.py` (загляните в него только после своей попытки).
Верное решение портал отметит сам и откроет разбор (если он запущен через `python start.py`).

## Разбор

> Читайте после своей попытки — в портале разбор открывается, когда упражнение решено.

**Идея.** Lightning убирает из вашего кода цикл обучения, но не отменяет его: Trainer сам вызывает
`training_step`/`validation_step`, агрегирует `self.log` по эпохе (среднее, взвешенное по размеру батча),
а колбэки после валидации решают, сохранять ли чекпоинт и не пора ли остановиться. Понимание этой семантики
объясняет «странные» эффекты: почему лучший чекпоинт не совпадает с эпохой, где сработал EarlyStopping, и
почему метрика эпохи не равна среднему по батчам.

**Типичные ошибки:**
- Учли sanity check как эпоху 0 — меняется val_loss эпохи 0.
- Невзвешенное среднее по батчам — последний маленький батч весит столько же, сколько полный.
- Обновляли best у EarlyStopping при улучшении меньше min_delta — wait сбрасывается, остановка уезжает.
- Обработали все 15 эпох из файла, хотя обучение остановилось раньше.
- Нумерация эпох с 1 — в Lightning (и в именах чекпоинтов `epoch=…`) счёт с 0.

**Что вынести:** `EarlyStopping(patience, min_delta)` смотрит на «значимые» улучшения, `ModelCheckpoint(save_top_k)`
— на любые; после `fit` используйте `trainer.test(ckpt_path='best')`, а не последние веса.
→ шпаргалка «PyTorch Lightning».
