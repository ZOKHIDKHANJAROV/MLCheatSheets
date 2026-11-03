# TensorFlow/Keras полный гайд

<!-- время: 18 мин -->

> **О чём лекция.** Как строить и обучать модели в Keras: три способа описать модель, что делают `compile` и `fit`, какие колбэки нужны всегда, как подготовить данные через `tf.data` и как сохранить модель для продакшена.

## Зачем это нужно

TensorFlow — фреймворк глубокого обучения от Google, Keras — его высокоуровневый интерфейс. Если в PyTorch цикл обучения вы обычно пишете сами, то в Keras типичная задача решается тремя вызовами: описать модель, `compile`, `fit`. Это делает Keras удобным для быстрых экспериментов и обучения новичков.

Сильная сторона экосистемы TensorFlow — развёртывание: TF Serving для серверов, TensorFlow Lite для телефонов и микроконтроллеров, TensorFlow.js для браузера. Начиная с Keras 3, тот же код Keras может работать и поверх PyTorch или JAX.

## Интуиция: модель как конструктор

Keras-модель собирается из слоёв, как из деталей конструктора. Вы описываете *что* должно быть в модели, а *как* её обучать — с какими функцией потерь, оптимизатором и метриками — задаёте отдельно в `compile`. Сам процесс обучения — батчи, эпохи, валидация, логирование — берёт на себя `fit`.

## Три способа описать модель

**Sequential API** — стопка слоёв, где выход одного идёт на вход следующего. Подходит для большинства простых моделей:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    keras.Input(shape=(20,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax"),
])
model.summary()
```

**Functional API** — граф слоёв: несколько входов и выходов, ветвления, skip-связи:

```python
inputs = keras.Input(shape=(20,))
h = layers.Dense(64, activation="relu")(inputs)
h = layers.Dense(64, activation="relu")(h) + h          # skip-связь
outputs = layers.Dense(3, activation="softmax")(h)
model = keras.Model(inputs, outputs)
```

**Наследование от `keras.Model`** — полная свобода: слои в `__init__`, логика в `call`. Это похоже на `nn.Module` в PyTorch и нужно для нестандартной логики прямого прохода.

Правило выбора: Sequential, пока хватает; Functional, когда нужны ветвления; подкласс — когда в прямом проходе нужна произвольная Python-логика.

## compile: как учить

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

`compile` связывает модель с оптимизатором, функцией потерь и метриками. Частая путаница — две кросс-энтропии:

- `sparse_categorical_crossentropy` — метки как номера классов: `[0, 2, 1]`;
- `categorical_crossentropy` — метки как one-hot векторы: `[[1,0,0], [0,0,1], …]`.

Математически это одно и то же, отличается только формат меток. Для бинарной задачи — `binary_crossentropy` и один выход с сигмоидой. Если выход модели без активации (логиты), укажите `from_logits=True` в функции потерь — так численно устойчивее.

## fit и колбэки

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=100, batch_size=64, callbacks=callbacks)
```

Колбэки — функции, которые Keras вызывает в определённые моменты обучения (конец эпохи, конец батча). Три самых полезных:

- **EarlyStopping** — останавливает обучение, когда валидационные потери перестали улучшаться, и с `restore_best_weights=True` возвращает лучшие веса;
- **ModelCheckpoint** — сохраняет лучшую модель на диск;
- **ReduceLROnPlateau** — уменьшает learning rate на плато.

`history.history` — словарь с потерями и метриками по эпохам; по нему строят кривые обучения. Ещё полезен колбэк `TensorBoard` для визуализации.

## tf.data: конвейер данных

Когда данные не помещаются в память или их нужно аугментировать на лету, используют `tf.data.Dataset`:

```python
import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds = ds.shuffle(10_000).batch(64).prefetch(tf.data.AUTOTUNE)
model.fit(ds, epochs=10)
```

`prefetch` готовит следующий батч, пока GPU считает текущий; `map(..., num_parallel_calls=tf.data.AUTOTUNE)` распараллеливает предобработку. Без этого GPU часто простаивает в ожидании данных.

## Свой цикл обучения

Если `fit` не подходит (GAN, нестандартные потери), шаг обучения пишут вручную через `GradientTape` — она записывает операции для автодифференцирования, как autograd в PyTorch:

```python
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(1e-3)

@tf.function                                  # компиляция в граф — быстрее
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

## Сохранение

```python
model.save("model.keras")                     # архитектура + веса + состояние оптимизатора
model = keras.models.load_model("model.keras")
model.export("serving_model")                 # SavedModel для TF Serving (Keras 3)
```

Формат `.keras` — стандарт для Keras 3; старый формат `.h5` поддерживается для совместимости.

## Подводные камни

- **Не тот формат меток**: one-hot с `sparse_categorical_crossentropy` или номера классов с `categorical_crossentropy` дают ошибку размерностей или бессмысленное обучение.
- **`validation_split` берёт последние строки** массива, без перемешивания. Если данные отсортированы (например, по классу), валидация будет нерепрезентативной — перемешайте заранее или передайте `validation_data`.
- **`training=True/False`** в своём цикле обязателен для dropout и BatchNorm — аналог `model.train()/eval()` в PyTorch.
- **Медленный конвейер данных.** Без `prefetch` и параллельного `map` обучение упирается в чтение данных, а не в GPU.
- **Устаревшие примеры.** В интернете много кода TF 1.x (`tf.Session`, `placeholder`) — в TF 2 он не нужен; также `tf.test.is_gpu_available()` устарел — используйте `tf.config.list_physical_devices("GPU")`.

## Итоги

- Keras: описать модель → `compile` (оптимизатор, потери, метрики) → `fit`.
- Sequential — для стопки слоёв, Functional — для графов с ветвлениями, подкласс `keras.Model` — для произвольной логики.
- `sparse_categorical_crossentropy` ждёт номера классов, `categorical_crossentropy` — one-hot.
- EarlyStopping, ModelCheckpoint и ReduceLROnPlateau стоит включать почти всегда.
- `tf.data` с `shuffle/batch/prefetch` держит GPU загруженным; `GradientTape` — для своего цикла обучения.

## Проверьте себя

1. Когда Sequential API недостаточно и нужен Functional API?
2. Какую функцию потерь выбрать, если метки — номера классов 0, 1, 2?
3. Что делает `EarlyStopping` с параметром `restore_best_weights=True`?
