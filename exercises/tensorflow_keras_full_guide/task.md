# Упражнение: Колбэки Keras: EarlyStopping и ReduceLROnPlateau по истории обучения

## Задание

Модель Keras учится с колбэками

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_delta=0.005,
                                      min_lr=1e-4),
]
model.fit(..., epochs=40, callbacks=callbacks)    # начальный learning rate 0.001
```

В `input.csv` — значения `loss` и `val_loss`, которые получились бы на эпохах 1…40 (для упражнения считаем,
что они не зависят от lr). Воспроизведите, что сделают колбэки. В конце каждой эпохи (сначала EarlyStopping,
затем ReduceLROnPlateau):

**EarlyStopping** (best = +∞, wait = 0 в начале): wait += 1; если val_loss < best − min_delta, то best = val_loss,
best_epoch = эпоха, wait = 0; иначе, если wait ≥ patience, — остановить обучение после этой эпохи.

**ReduceLROnPlateau** (свой best = +∞, свой wait = 0): если val_loss < best − min_delta, то best = val_loss,
wait = 0; иначе wait += 1, и если wait ≥ patience: если текущий lr > min_lr, то
lr = max(lr · factor, min_lr) и wait = 0 (если lr уже равен min_lr — ничего не происходит, wait не сбрасывается).
Новый lr действует со следующей эпохи.

## Подсказки

- У колбэков разные min_delta: падение val_loss на 0.002 — улучшение для EarlyStopping, но не для ReduceLROnPlateau.
- Колонка `lr` — learning rate, с которым **шла** эпоха (именно его Keras пишет в `history`), уменьшение
  видно только в следующей строке.
- Оформите колбэки классами с `on_epoch_end(epoch, val_loss)`; когда EarlyStopping вернёт «стоп», выйдите из цикла.
  `best_epoch` — эпоха, веса которой вернёт `restore_best_weights=True`.

## Ожидаемый формат выходного файла

CSV с колонками `epoch, val_loss, lr, es_wait, rlr_wait, best_epoch` — по строке на каждую пройденную эпоху
(с 1 до эпохи остановки включительно); счётчики и `best_epoch` — после обработки эпохи. `lr` округлить
до 6 знаков.

## Проверка

Сохраните свой результат в `my_answer.csv` (в папке упражнения) и выполните из корня репозитория:

```bash
python tools/check_exercise.py exercises/tensorflow_keras_full_guide
```

Эталонное решение с пояснениями — `solution.py` (загляните в него только после своей попытки).
Верное решение портал отметит сам и откроет разбор (если он запущен через `python start.py`).

## Разбор

> Читайте после своей попытки — в портале разбор открывается, когда упражнение решено.

**Идея.** Колбэки — это маленькие конечные автоматы, которые Keras вызывает в конце каждой эпохи. EarlyStopping
считает эпохи без улучшения и останавливает обучение, `restore_best_weights=True` возвращает веса лучшей эпохи.
ReduceLROnPlateau на плато уменьшает шаг: часто после этого val_loss снова начинает падать. Их счётчики
независимы, а разные `min_delta` дают разные представления о том, что такое «улучшение».

**Типичные ошибки:**
- Записывали в `lr` уже уменьшенное значение — история сдвигается на эпоху.
- Один общий счётчик wait для двух колбэков или общий best.
- Продолжали уменьшать lr ниже `min_lr` или сбрасывали wait, когда lr уже на минимуме.
- Не остановились на эпохе EarlyStopping и обработали все 40 строк.
- Забыли `restore_best_weights=True` в реальном коде: модель после `fit` — с весами последней, а не лучшей эпохи.

**Что вынести:** patience EarlyStopping должен быть больше patience ReduceLROnPlateau — иначе шаг не успеет
уменьшиться до остановки. `history.history` хранит и `learning_rate` — по нему видно, когда сработал колбэк.
→ шпаргалка «TensorFlow / Keras: полное руководство».
