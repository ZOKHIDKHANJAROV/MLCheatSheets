#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений по основам глубокого обучения (Путь 2) — только NumPy, без PyTorch.

Цель — посчитать «руками» то, что фреймворки делают внутри: активации и их производные,
softmax и кросс-энтропию, обратное распространение, прямой проход MLP, оптимизаторы,
свёртку с пулингом и self-attention. Ответы проверяются так же, как остальные упражнения:
    python tools/check_exercise.py exercises/<имя>

Запуск из корня репозитория:
    python tools/generate_dl_exercises.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build

# ---------------------------------------------------------------- данные

def data_activations(rng):
    x = np.concatenate([[-4, -1, 0, 0.5, 3], rng.uniform(-4, 4, 15)]).round(2)
    return pd.DataFrame({"x": x})


def data_softmax(rng):
    n = 12
    z = rng.normal(0, 2, (n, 3)).round(2)
    z[3] = [1000.0, 999.0, 995.0]       # большие логиты: наивная формула переполнится
    z[7] = [-800.0, -805.0, -790.0]
    y = rng.integers(0, 3, n)
    y[3] = 2                            # заведомо «уверенно неверный» прогноз
    return pd.DataFrame({"id": np.arange(1, n + 1), "z0": z[:, 0], "z1": z[:, 1], "z2": z[:, 2], "y": y})


def data_neuron(rng):
    n = 10
    return pd.DataFrame({"id": np.arange(1, n + 1), "x1": rng.uniform(-2, 2, n).round(2),
                         "x2": rng.uniform(-2, 2, n).round(2), "y": rng.integers(0, 2, n)})


def data_mlp(rng):
    n = 15
    inputs = pd.DataFrame({"id": np.arange(1, n + 1), "x1": rng.uniform(-1, 1, n).round(2), "x2": rng.uniform(-1, 1, n).round(2)})
    weights = {
        "W1": rng.normal(0, 1, (3, 2)).round(2).tolist(),   # скрытый слой: 3 нейрона × 2 входа
        "b1": rng.normal(0, 0.5, 3).round(2).tolist(),
        "W2": rng.normal(0, 1, (1, 3)).round(2).tolist(),   # выход: 1 нейрон × 3 скрытых
        "b2": rng.normal(0, 0.5, 1).round(2).tolist(),
    }
    return {"input.csv": inputs, "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_optimizers(rng):
    return pd.DataFrame([
        {"optimizer": "sgd", "lr": 0.05, "beta1": 0.0, "beta2": 0.0, "eps": 0.0},
        {"optimizer": "momentum", "lr": 0.02, "beta1": 0.9, "beta2": 0.0, "eps": 0.0},
        {"optimizer": "adam", "lr": 0.1, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    ])


def data_convolution(rng):
    img = np.zeros((7, 7), dtype=int)
    img[:, :3] = 5                                           # вертикальная граница: слева светло, справа темно
    img += rng.integers(0, 3, (7, 7))                        # немного шума
    return pd.DataFrame(img, columns=[f"c{j}" for j in range(7)])


def data_attention(rng):
    n, d = 4, 3
    m = {name: rng.normal(0, 1, (n, d)).round(2) for name in "qkv"}
    cols = {"token": ["Я", "люблю", "машинное", "обучение"]}
    for name in "qkv":
        for j in range(d):
            cols[f"{name}{j}"] = m[name][:, j]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "activation_functions",
        "title": "Функции активации и их производные",
        "data": data_activations,
        "task": """
            Дан файл `input.csv` со значениями `x`. Для каждого x вычислите функции активации и их производные:

            1. **sigmoid**: σ(x) = 1 / (1 + e⁻ˣ), производная σ′(x) = σ(x)·(1 − σ(x))
            2. **tanh**: tanh(x), производная 1 − tanh²(x)
            3. **ReLU**: max(0, x), производная 1 при x > 0, иначе 0 (в точке 0 принимаем 0)

            ## Подсказки

            - Используйте NumPy и векторные операции — без циклов.
            - Выразите производные через уже посчитанные значения функций — так делают и во фреймворках.
            - Посмотрите на максимум σ′(x): почему сигмоида приводит к затуханию градиента в глубоких сетях?

            ## Ожидаемый формат выходного файла

            CSV с колонками `x, sigmoid, d_sigmoid, tanh, d_tanh, relu, d_relu` в исходном порядке строк,
            значения округлить до 4 знаков.
        """,
        "solution": """
            def main():
                x = pd.read_csv('input.csv')['x'].to_numpy()

                sigmoid = 1 / (1 + np.exp(-x))
                tanh = np.tanh(x)
                relu = np.maximum(0, x)

                result = pd.DataFrame({
                    'x': x,
                    'sigmoid': sigmoid, 'd_sigmoid': sigmoid * (1 - sigmoid),   # максимум 0.25 при x = 0
                    'tanh': tanh, 'd_tanh': 1 - tanh ** 2,                      # максимум 1 при x = 0
                    'relu': relu, 'd_relu': (x > 0).astype(float),              # не затухает при x > 0
                })
                print("max σ′ =", result['d_sigmoid'].max().round(4), "  max tanh′ =", result['d_tanh'].max().round(4))
                return result.round(4)
        """,
    },
    {
        "name": "loss_functions",
        "title": "Softmax и кросс-энтропия (численно устойчиво)",
        "data": data_softmax,
        "task": """
            Дан файл `input.csv` с логитами модели для трёх классов (`z0, z1, z2`) и истинным классом `y`.

            1. Вычислите вероятности классов **softmax**: pₖ = e^zₖ / Σⱼ e^zⱼ
            2. Вычислите **кросс-энтропию** для каждого объекта: loss = −log p_y
            3. Определите предсказанный класс `pred` — argmax вероятностей

            Внимание: в строках 4 и 8 логиты очень большие по модулю (±1000). Наивная формула даст переполнение
            (`inf`, `nan`). Используйте устойчивые приёмы:

            - softmax: вычтите максимум логита в строке перед экспонентой — результат не изменится;
            - loss: −log p_y = logsumexp(z) − z_y, где logsumexp(z) = max(z) + log Σ e^(z − max(z)).

            ## Подсказки

            - Попробуйте сначала наивную формулу на строке 4 и посмотрите, что получится.
            - В PyTorch именно поэтому `nn.CrossEntropyLoss` принимает логиты, а не вероятности.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, p0, p1, p2, loss, pred`; вероятности и loss округлить до 4 знаков.
        """,
        "solution": """
            def main():
                df = pd.read_csv('input.csv')
                z = df[['z0', 'z1', 'z2']].to_numpy()
                y = df['y'].to_numpy()

                # Устойчивый softmax: сдвиг на максимум не меняет результат, но убирает переполнение
                shifted = z - z.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                p = exp / exp.sum(axis=1, keepdims=True)

                # Устойчивая кросс-энтропия через log-sum-exp
                logsumexp = z.max(axis=1) + np.log(exp.sum(axis=1))
                loss = logsumexp - z[np.arange(len(y)), y]

                with np.errstate(over='ignore', invalid='ignore'):
                    naive = np.exp(z[3]) / np.exp(z[3]).sum()
                print("наивный softmax для строки 4:", naive, " устойчивый:", p[3].round(4))

                return pd.DataFrame({'id': df['id'], 'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2],
                                     'loss': loss, 'pred': p.argmax(axis=1)}).round(4)
        """,
    },
    {
        "name": "backpropagation",
        "title": "Обратное распространение для одного нейрона",
        "data": data_neuron,
        "task": """
            Нейрон с сигмоидой: a = σ(z), z = w₁·x₁ + w₂·x₂ + b, где **w₁ = 0.5, w₂ = −0.3, b = 0.1**.
            Функция потерь — бинарная кросс-энтропия: L = −[y·log a + (1 − y)·log(1 − a)].

            Для каждой строки `input.csv` (признаки `x1, x2`, метка `y`) вычислите:

            1. **Прямой проход**: z, a, L
            2. **Обратный проход** по цепному правилу:
               - ∂L/∂z — для сигмоиды с кросс-энтропией это упрощается до a − y (выведите это сами!)
               - ∂L/∂w₁ = ∂L/∂z · x₁, ∂L/∂w₂ = ∂L/∂z · x₂, ∂L/∂b = ∂L/∂z

            ## Подсказки

            - ∂L/∂a = −y/a + (1 − y)/(1 − a), ∂a/∂z = a(1 − a). Перемножьте и упростите.
            - Проверьте градиент численно: (L(w₁ + ε) − L(w₁ − ε)) / 2ε при ε = 1e−5 должно совпасть с ∂L/∂w₁.
            - Средний градиент по всем строкам — это то, что использует градиентный спуск на батче.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, z, a, loss, dz, dw1, dw2, db`, значения округлить до 4 знаков.
        """,
        "solution": """
            W1, W2, B = 0.5, -0.3, 0.1


            def forward(x1, x2, y, w1=W1, w2=W2, b=B):
                z = w1 * x1 + w2 * x2 + b
                a = 1 / (1 + np.exp(-z))
                loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
                return z, a, loss


            def main():
                df = pd.read_csv('input.csv')
                x1, x2, y = df['x1'].to_numpy(), df['x2'].to_numpy(), df['y'].to_numpy()

                z, a, loss = forward(x1, x2, y)
                dz = a - y                      # ∂L/∂a · ∂a/∂z = (a − y) / (a(1 − a)) · a(1 − a)
                dw1, dw2, db = dz * x1, dz * x2, dz

                # Численная проверка градиента по w1
                eps = 1e-5
                numeric = (forward(x1, x2, y, w1=W1 + eps)[2] - forward(x1, x2, y, w1=W1 - eps)[2]) / (2 * eps)
                print("max |аналитический − численный| для dw1:", np.abs(numeric - dw1).max())

                return pd.DataFrame({'id': df['id'], 'z': z, 'a': a, 'loss': loss,
                                     'dz': dz, 'dw1': dw1, 'dw2': dw2, 'db': db}).round(4)
        """,
    },
    {
        "name": "mlp",
        "title": "Прямой проход многослойного перцептрона",
        "data": data_mlp,
        "task": """
            Дана обученная сеть 2 → 3 → 1. Веса лежат в `weights.json`:

            - `W1` — матрица 3×2 (строка = нейрон скрытого слоя), `b1` — 3 смещения, активация **tanh**;
            - `W2` — матрица 1×3, `b2` — 1 смещение, выход — **sigmoid**.

            Для каждой строки `input.csv` (`x1, x2`) вычислите:

            1. Скрытый слой: h = tanh(W1·x + b1) → `h1, h2, h3`
            2. Выход: p = σ(W2·h + b2) → вероятность класса 1
            3. Предсказание `pred` = 1, если p ≥ 0.5, иначе 0

            ## Подсказки

            - Обработайте все строки разом матрично: H = tanh(X·W1ᵀ + b1), где X — матрица n×2.
            - `json.load(open('weights.json'))` → словарь со списками; превратите их в `np.array`.
            - Сколько параметров у этой сети? (Ответ: 3·2 + 3 + 1·3 + 1 = 13.)

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, h1, h2, h3, p, pred`; значения округлить до 4 знаков.
        """,
        "solution": """
            import json


            def main():
                df = pd.read_csv('input.csv')
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                X = df[['x1', 'x2']].to_numpy()                     # n × 2

                H = np.tanh(X @ w['W1'].T + w['b1'])                # n × 3 — скрытый слой
                p = 1 / (1 + np.exp(-(H @ w['W2'].T + w['b2'])))    # n × 1 — выход
                p = p[:, 0]

                n_params = sum(v.size for v in w.values())
                print("параметров в сети:", n_params)
                return pd.DataFrame({'id': df['id'], 'h1': H[:, 0], 'h2': H[:, 1], 'h3': H[:, 2],
                                     'p': p, 'pred': (p >= 0.5).astype(int)}).round(4)
        """,
    },
    {
        "name": "optimizers",
        "title": "SGD, Momentum и Adam на «овраге»",
        "data": data_optimizers,
        "task": """
            Минимизируйте функцию f(x, y) = x² + 10·y² (вытянутый «овраг», минимум в (0, 0))
            тремя оптимизаторами с параметрами из `input.csv`. Старт — точка **(−2, 1.5)**, ровно **50 шагов**.
            Градиент: ∇f = (2x, 20y).

            Формулы (θ — вектор (x, y), g — градиент в текущей точке, t — номер шага с 1):

            - **sgd**: θ ← θ − lr·g
            - **momentum**: v ← β₁·v + g, θ ← θ − lr·v (v₀ = 0)
            - **adam**: m ← β₁·m + (1 − β₁)·g, s ← β₂·s + (1 − β₂)·g²,
              m̂ = m / (1 − β₁ᵗ), ŝ = s / (1 − β₂ᵗ), θ ← θ − lr·m̂ / (√ŝ + eps) (m₀ = s₀ = 0)

            ## Подсказки

            - Сохраняйте траекторию и нарисуйте её поверх линий уровня f — сравните пути оптимизаторов.
            - Почему SGD с lr = 0.11 расходится? (Подсказка: устойчивость по y требует lr < 2/20.)
            - То же самое можно увидеть в интерактиве «Градиентный спуск» в портале.
            - Не удивляйтесь, если SGD окажется ближе к минимуму: на гладкой квадратичной функции с хорошо подобранным lr
              он очень эффективен. Momentum «проскакивает» минимум по инерции, а Adam делает шаги ~lr по каждой оси
              и колеблется вблизи минимума. Преимущества Adam видны на шумных градиентах и плохо масштабированных параметрах.

            ## Ожидаемый формат выходного файла

            CSV с колонками `optimizer, x, y, loss` — по строке на оптимизатор в порядке input.csv;
            x, y, loss после 50 шагов, округлить до 4 знаков.
        """,
        "solution": """
            f = lambda th: th[0] ** 2 + 10 * th[1] ** 2
            grad = lambda th: np.array([2 * th[0], 20 * th[1]])


            def run(name, lr, beta1, beta2, eps, steps=50):
                th = np.array([-2.0, 1.5])
                v = m = s = np.zeros(2)
                for t in range(1, steps + 1):
                    g = grad(th)
                    if name == 'sgd':
                        th = th - lr * g
                    elif name == 'momentum':
                        v = beta1 * v + g
                        th = th - lr * v
                    elif name == 'adam':
                        m = beta1 * m + (1 - beta1) * g
                        s = beta2 * s + (1 - beta2) * g ** 2
                        m_hat, s_hat = m / (1 - beta1 ** t), s / (1 - beta2 ** t)
                        th = th - lr * m_hat / (np.sqrt(s_hat) + eps)
                return th


            def main():
                rows = []
                for _, r in pd.read_csv('input.csv').iterrows():
                    th = run(r['optimizer'], r['lr'], r['beta1'], r['beta2'], r['eps'])
                    rows.append({'optimizer': r['optimizer'], 'x': th[0], 'y': th[1], 'loss': f(th)})
                    print(f"{r['optimizer']:>9}: ({th[0]:+.4f}, {th[1]:+.4f})  f = {f(th):.6f}")
                return pd.DataFrame(rows).round(4)
        """,
    },
    {
        "name": "cnn_basics",
        "title": "Свёртка, ReLU и max pooling вручную",
        "data": data_convolution,
        "task": """
            Файл `input.csv` — изображение 7×7 (строки — ряды пикселей, колонки `c0…c6`).
            Слева светлая часть, справа тёмная — вертикальная граница.

            1. Примените свёртку с ядром — детектором вертикальных границ:

               ```
               [[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]]
               ```

               без дополнения нулями (padding = 0) и с шагом 1. Как во всех DL-фреймворках, ядро **не переворачивается**
               (это кросс-корреляция): out[i, j] = Σₘ Σₙ img[i + m, j + n] · K[m, n]. Размер выхода: 5×5.
            2. Примените **ReLU** к результату.
            3. Примените **max pooling** 2×2 с шагом 2 к карте после ReLU. Неполные окна (последняя строка и столбец)
               отбрасываются, как в `nn.MaxPool2d` по умолчанию, — получится 2×2.

            ## Подсказки

            - Размер выхода свёртки: (7 − 3) / 1 + 1 = 5.
            - Где отклик сильнее всего и какого он знака? Ядро вычитает правый столбец окна из левого:
              слева светлее — положительный отклик, слева темнее — отрицательный (и его обнулит ReLU).
            - Сделайте сначала через два цикла, потом — через `numpy.lib.stride_tricks.sliding_window_view`.

            ## Ожидаемый формат выходного файла

            CSV в «длинном» формате с колонками `stage, i, j, value`:
            сначала все 25 значений `conv` (по строкам: i — строка, j — столбец), затем 25 значений `relu`,
            затем 4 значения `pool`. Значения — целые числа.
        """,
        "solution": """
            from numpy.lib.stride_tricks import sliding_window_view

            KERNEL = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])


            def main():
                img = pd.read_csv('input.csv').to_numpy()

                # Свёртка (кросс-корреляция): окно 3×3 скользит по изображению
                windows = sliding_window_view(img, (3, 3))                    # 5 × 5 × 3 × 3
                conv = (windows * KERNEL).sum(axis=(2, 3))
                relu = np.maximum(conv, 0)
                h = relu.shape[0] // 2 * 2                                    # отбрасываем неполные окна
                pool = relu[:h, :h].reshape(h // 2, 2, h // 2, 2).max(axis=(1, 3))

                print("свёртка:\\n", conv, "\\nmax pooling:\\n", pool)
                rows = []
                for stage, m in [('conv', conv), ('relu', relu), ('pool', pool)]:
                    for i in range(m.shape[0]):
                        for j in range(m.shape[1]):
                            rows.append({'stage': stage, 'i': i, 'j': j, 'value': int(m[i, j])})
                return pd.DataFrame(rows)
        """,
    },
    {
        "name": "self_attention",
        "title": "Scaled dot-product attention",
        "data": data_attention,
        "task": """
            В файле `input.csv` — 4 токена и уже посчитанные для них векторы запроса, ключа и значения
            (размерность d = 3): колонки `q0..q2`, `k0..k2`, `v0..v2`.

            Вычислите self-attention:

            1. Оценки сходства: S = Q·Kᵀ / √d (матрица 4×4)
            2. Веса внимания: A = softmax(S) **по строкам** (каждая строка — распределение по всем токенам)
            3. Выход: O = A·V (матрица 4×3)

            ## Подсказки

            - Не забудьте деление на √d и устойчивый softmax (вычитание максимума строки).
            - Сумма каждой строки A должна быть равна 1 — проверьте.
            - Бонус: примените каузальную маску (токен i видит только токены ≤ i): до softmax поставьте −∞
              в верхний треугольник S. Так работает декодер GPT. В ответ это не входит.

            ## Ожидаемый формат выходного файла

            CSV с колонками `token, w0, w1, w2, w3, o0, o1, o2`: веса внимания токена ко всем четырём токенам
            и его выходной вектор. Значения округлить до 4 знаков.
        """,
        "solution": """
            def softmax(s):
                e = np.exp(s - s.max(axis=-1, keepdims=True))
                return e / e.sum(axis=-1, keepdims=True)


            def main():
                df = pd.read_csv('input.csv')
                Q = df[['q0', 'q1', 'q2']].to_numpy()
                K = df[['k0', 'k1', 'k2']].to_numpy()
                V = df[['v0', 'v1', 'v2']].to_numpy()
                d = Q.shape[1]

                A = softmax(Q @ K.T / np.sqrt(d))       # 4 × 4, строки суммируются в 1
                O = A @ V                               # 4 × 3
                assert np.allclose(A.sum(axis=1), 1)

                # Бонус: каузальная маска (как в декодере GPT)
                masked = np.where(np.triu(np.ones((4, 4)), k=1) == 1, -np.inf, Q @ K.T / np.sqrt(d))
                print("веса с каузальной маской:\\n", softmax(masked).round(3))

                result = pd.DataFrame({'token': df['token']})
                for j in range(4):
                    result[f'w{j}'] = A[:, j]
                for j in range(3):
                    result[f'o{j}'] = O[:, j]
                return result.round(4)
        """,
    },
]


def main():
    print("Генерация упражнений по Deep Learning:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=201):
        build(exercise, seed)


if __name__ == "__main__":
    main()
