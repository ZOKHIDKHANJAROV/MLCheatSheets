# PyTorch полный гайд

<!-- время: 20 мин -->

> **О чём лекция.** Как устроен PyTorch: тензоры, автоматическое дифференцирование, модули, Dataset и DataLoader, и главное — стандартный цикл обучения, который вы будете писать в каждом проекте.

## Зачем это нужно

PyTorch — самый популярный фреймворк глубокого обучения в исследованиях и всё шире в продакшене. На нём написаны Hugging Face Transformers, большинство современных моделей компьютерного зрения и генеративных моделей. Он ценится за то, что код выглядит как обычный Python: модель — класс, обучение — обычный цикл `for`, отладка — обычный `print` и отладчик.

Хорошая новость: весь PyTorch для старта — это пять понятий. Тензоры, autograd, `nn.Module`, `DataLoader` и цикл обучения. Остальное — вариации.

## Интуиция: NumPy с градиентами и GPU

Тензор PyTorch — это почти массив NumPy, у которого есть две суперспособности:

- он может жить на **GPU** и считаться там в десятки раз быстрее;
- он может **запоминать операции**, которые с ним делали, чтобы потом автоматически посчитать градиенты.

Граф вычислений строится *динамически* — по ходу выполнения обычного Python-кода. Поэтому в модели можно использовать `if`, циклы и любую логику.

## Тензоры

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
z = torch.zeros(3, 4)
r = torch.randn(2, 3)                  # нормальное распределение
print(x.shape, x.dtype)                # torch.Size([2, 2]) torch.float32

y = x @ x.T                            # матричное умножение
a = torch.from_numpy(np.arange(3.0))   # из NumPy (общая память!), нужен import numpy as np
b = x.numpy()                          # в NumPy (только для тензоров на CPU)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                       # перенос на GPU
```

Важные детали: по умолчанию числа с плавающей точкой — `float32`, целые — `int64`; метки классов для `CrossEntropyLoss` должны быть именно `int64` (`torch.long`). Операции между тензорами на разных устройствах невозможны: модель и данные должны быть на одном `device`.

## Autograd: автоматические градиенты

Если у тензора `requires_grad=True`, PyTorch записывает все операции с ним. Вызов `.backward()` на скалярном результате применяет цепное правило и кладёт градиенты в поле `.grad`:

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (w * 2 - 4) ** 2               # (2w − 4)²
loss.backward()
print(w.grad)                          # tensor(8.) = 2·(2·3 − 4)·2
```

Параметры слоёв (`nn.Linear` и др.) создаются с `requires_grad=True` автоматически. Градиенты **накапливаются**: каждый `backward()` прибавляет к `.grad`, а не перезаписывает. Поэтому перед каждым шагом их обнуляют.

## Модели: nn.Module

Модель — класс-наследник `nn.Module`. В `__init__` создаются слои, в `forward` описывается прямой проход:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_classes),   # на выходе логиты, без softmax
        )

    def forward(self, x):
        return self.net(x)

model = MLP(20, 64, 3).to(device)
print(sum(p.numel() for p in model.parameters()))   # число параметров
```

Модель вызывают как функцию — `model(x)`, а не `model.forward(x)`: так срабатывают служебные хуки.

## Данные: Dataset и DataLoader

`Dataset` отвечает на два вопроса: сколько объектов (`__len__`) и как получить i-й (`__getitem__`). `DataLoader` собирает объекты в батчи, перемешивает и может грузить данные в несколько процессов.

```python
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(torch.randn(1000, 20), torch.randint(0, 3, (1000,)))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

val_ds = TensorDataset(torch.randn(200, 20), torch.randint(0, 3, (200,)))
val_loader = DataLoader(val_ds, batch_size=256)        # валидацию не перемешивают
```

Для своих данных (картинки с диска, тексты) пишут класс-наследник `Dataset`; тяжёлую загрузку делают в `__getitem__`, чтобы не держать всё в памяти.

## Цикл обучения

Это сердце любого проекта на PyTorch. Одна итерация — пять шагов:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()                              # включает dropout, обучение BatchNorm
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                  # 1. обнулить старые градиенты
        logits = model(xb)                     # 2. прямой проход
        loss = criterion(logits, yb)           # 3. потери
        loss.backward()                        # 4. обратный проход
        optimizer.step()                       # 5. шаг оптимизатора

    model.eval()                               # выключает dropout, BatchNorm — по накопленной статистике
    with torch.no_grad():                      # граф не строится: быстрее и меньше памяти
        correct = sum((model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
                      for xb, yb in val_loader)
    print(epoch, loss.item(), correct / len(val_loader.dataset))
```

`model.train()` и `model.eval()` переключают поведение слоёв: dropout на обучении случайно обнуляет нейроны, а на оценке — нет; BatchNorm на обучении нормирует по статистике батча, а на оценке — по накопленной. `torch.no_grad()` отключает построение графа, когда градиенты не нужны.

## Сохранение и загрузка

Сохраняют не весь объект, а словарь весов:

```python
torch.save(model.state_dict(), "model.pt")

model = MLP(20, 64, 3)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

Чтобы продолжить обучение, в чекпоинт кладут ещё `optimizer.state_dict()` и номер эпохи.

## Подводные камни

- **Забытый `zero_grad()`** — градиенты батчей складываются, обучение идёт неправильно, без всякой ошибки.
- **Забытый `model.eval()`** при оценке — dropout продолжает работать, метрики «скачут» и занижены.
- **Разные устройства.** `RuntimeError: Expected all tensors to be on the same device` — переносите и модель, и каждый батч на один `device`.
- **Softmax перед `CrossEntropyLoss`.** Функция ждёт логиты; метки — `long` с номерами классов, а не one-hot.
- **`loss` вместо `loss.item()` в логах.** Накопление тензоров с графом в списке держит в памяти весь граф; для логирования берите `.item()`.
- **Воспроизводимость.** Фиксируйте `torch.manual_seed(...)`; полная детерминированность на GPU требует дополнительных настроек.

## Итоги

- Тензор — массив NumPy с поддержкой GPU и автоматических градиентов.
- Autograd записывает операции и по `backward()` считает градиенты; градиенты накапливаются.
- Модель — `nn.Module` со слоями в `__init__` и логикой в `forward`.
- `Dataset` описывает данные, `DataLoader` режет их на перемешанные батчи.
- Цикл обучения: `zero_grad` → прямой проход → потери → `backward` → `step`; для оценки — `eval()` и `no_grad()`.

## Проверьте себя

1. Перечислите пять шагов одной итерации обучения в PyTorch.
2. Чем отличается поведение модели в режимах `train()` и `eval()`?
3. Зачем оборачивать оценку модели в `torch.no_grad()`?
