#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений для этапов учебных путей, где не хватало практики.

Путь 1 (новичок): градиентный спуск, k-NN, наивный Байес, ROC-кривая, случайный лес.
Путь 2 (Deep Learning, только NumPy — без PyTorch): цикл обучения «как в PyTorch», ячейка LSTM,
word embeddings, VAE (репараметризация, KL, ELBO), функции потерь GAN.
Путь 4 / CV: IoU и NMS для детекции, метрики сегментации (IoU, Dice, pixel accuracy).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_path_exercises.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build, split_column

# ---------------------------------------------------------------- данные


def data_gradient_descent(rng):
    n = 40
    area = rng.uniform(20, 120, n).round(1)
    age = rng.integers(0, 40, n)
    price = 3 + 0.05 * area - 0.04 * age + rng.normal(0, 0.4, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "area": area, "age": age, "price": price.round(2)})


def data_knn(rng):
    n = 50
    income = rng.integers(25, 151, n) * 1000                       # рубли: масштаб ~10⁵
    debt_ratio = rng.uniform(0, 1, n).round(2)                     # доля: масштаб ~1
    score = 3 * debt_ratio - income / 100_000 + rng.normal(0, 0.3, n)
    default = (score > 0.9).astype(int)
    return pd.DataFrame({"id": np.arange(1, n + 1), "income": income, "debt_ratio": debt_ratio,
                         "default": default, "split": split_column(n, 0.3, rng)})


def data_naive_bayes(rng):
    spam = ["скидка", "бесплатно", "выигрыш", "акция", "кредит", "срочно"]
    ham = ["встреча", "отчёт", "проект", "завтра", "обед", "код"]
    shared = ["привет", "сегодня", "вы"]
    rows = []
    labels = ["spam"] * 13 + ["ham"] * 13
    rng.shuffle(labels)
    for i, label in enumerate(labels):
        own, other = (spam, ham) if label == "spam" else (ham, spam)
        words = []
        for _ in range(rng.integers(4, 8)):
            u = rng.uniform()
            pool = own if u < 0.6 else (shared if u < 0.85 else other)
            words.append(pool[rng.integers(len(pool))])
        rows.append({"id": i + 1, "text": " ".join(words), "label": label})
    df = pd.DataFrame(rows)
    df["split"] = ["train"] * 20 + ["test"] * 6
    df.loc[22, "text"] += " кот"                                   # слово, которого нет в train
    return df


def data_roc_auc(rng):
    n = 20
    y = rng.permutation(np.repeat([0, 1], [11, 9]))
    score = np.clip(0.25 * y + rng.uniform(0, 0.75, n), 0, 1).round(1)  # грубое округление → есть совпадения
    return pd.DataFrame({"id": np.arange(1, n + 1), "y_true": y, "y_score": score})


def data_random_forest(rng):
    n = 240
    tenure = rng.integers(1, 72, n)
    monthly_charge = rng.uniform(20, 120, n).round(2)
    support_calls = rng.poisson(1.5, n)
    noise = rng.normal(0, 1, n).round(3)                           # признак-пустышка
    logit = -0.06 * tenure + 0.035 * monthly_charge + 0.7 * support_calls - 1.5
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"id": np.arange(1, n + 1), "tenure": tenure, "monthly_charge": monthly_charge,
                         "support_calls": support_calls, "noise": noise, "churn": churn,
                         "split": split_column(n, 0.25, rng)})


def data_training_loop(rng):
    n = 24
    x1 = rng.uniform(-1, 1, n).round(2)
    x2 = rng.uniform(-1, 1, n).round(2)
    y = (np.sin(2 * x1) + 0.5 * x2 + rng.normal(0, 0.05, n)).round(3)
    batch = np.repeat([1, 2, 3], 8)
    rng.shuffle(batch)                                            # фиксированное «перемешивание» — уже в данных
    inputs = pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1, "x2": x2, "y": y, "batch": batch})
    weights = {
        "W1": rng.normal(0, 0.7, (3, 2)).round(2).tolist(),        # nn.Linear(2, 3).weight: 3 × 2
        "b1": rng.normal(0, 0.1, 3).round(2).tolist(),
        "W2": rng.normal(0, 0.7, (1, 3)).round(2).tolist(),        # nn.Linear(3, 1).weight: 1 × 3
        "b2": [0.0],
    }
    return {"input.csv": inputs, "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_lstm(rng):
    T, D, H = 5, 2, 3
    x = rng.uniform(-1, 1, (T, D)).round(2)
    inputs = pd.DataFrame({"t": np.arange(1, T + 1), "x1": x[:, 0], "x2": x[:, 1]})
    weights = {
        "W_ih": rng.normal(0, 0.5, (4 * H, D)).round(2).tolist(),   # как nn.LSTM.weight_ih_l0: (4H × D)
        "W_hh": rng.normal(0, 0.5, (4 * H, H)).round(2).tolist(),   # weight_hh_l0: (4H × H)
        "b_ih": rng.normal(0, 0.2, 4 * H).round(2).tolist(),
        "b_hh": rng.normal(0, 0.2, 4 * H).round(2).tolist(),
    }
    return {"input.csv": inputs, "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_word_embeddings(rng):
    # оси смысла: королевский, мужской, молодой, фрукт, город, страна, Франция, Россия, Германия
    base = {
        "король": [1, 1, 0, 0, 0, 0, 0, 0, 0], "королева": [1, -1, 0, 0, 0, 0, 0, 0, 0],
        "принц": [1, 1, 1, 0, 0, 0, 0, 0, 0], "принцесса": [1, -1, 1, 0, 0, 0, 0, 0, 0],
        "мужчина": [0, 1, 0, 0, 0, 0, 0, 0, 0], "женщина": [0, -1, 0, 0, 0, 0, 0, 0, 0],
        "мальчик": [0, 1, 1, 0, 0, 0, 0, 0, 0], "девочка": [0, -1, 1, 0, 0, 0, 0, 0, 0],
        "яблоко": [0, 0, 0, 1, 0, 0, 0, 0, 0], "груша": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "банан": [0, 0, 0, 1, 0, 0, 0, 0, 0], "апельсин": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "париж": [0, 0, 0, 0, 1, 0, 1, 0, 0], "франция": [0, 0, 0, 0, 0, 1, 1, 0, 0],
        "москва": [0, 0, 0, 0, 1, 0, 0, 1, 0], "россия": [0, 0, 0, 0, 0, 1, 0, 1, 0],
        "берлин": [0, 0, 0, 0, 1, 0, 0, 0, 1], "германия": [0, 0, 0, 0, 0, 1, 0, 0, 1],
    }
    words = list(base)
    E = np.array([base[w] for w in words], dtype=float) + rng.normal(0, 0.08, (len(words), 9))
    table = pd.DataFrame(E.round(2), columns=[f"e{j}" for j in range(9)])
    table.insert(0, "word", words)
    queries = pd.DataFrame([
        {"query_id": 1, "kind": "neighbors", "a": "яблоко", "b": "", "c": ""},
        {"query_id": 2, "kind": "neighbors", "a": "москва", "b": "", "c": ""},
        {"query_id": 3, "kind": "neighbors", "a": "король", "b": "", "c": ""},
        {"query_id": 4, "kind": "analogy", "a": "король", "b": "мужчина", "c": "женщина"},
        {"query_id": 5, "kind": "analogy", "a": "париж", "b": "франция", "c": "россия"},
        {"query_id": 6, "kind": "analogy", "a": "принц", "b": "мальчик", "c": "девочка"},
    ])
    return {"input.csv": table, "queries.csv": queries}


def data_vae(rng):
    n = 6
    mu = rng.normal(0, 1, (n, 2)).round(2)
    logvar = rng.uniform(-2, 1, (n, 2)).round(2)
    mu[0], logvar[0] = 0.0, 0.0                                    # апостериор совпал с N(0, I): KL = 0
    eps = rng.normal(0, 1, (n, 2)).round(2)
    x = rng.integers(0, 2, (n, 4))
    x_hat = np.clip(np.where(x == 1, rng.uniform(0.5, 0.95, (n, 4)), rng.uniform(0.05, 0.5, (n, 4))), 0.05, 0.95).round(2)
    enc = pd.DataFrame({"id": np.arange(1, n + 1), "mu1": mu[:, 0], "mu2": mu[:, 1],
                        "logvar1": logvar[:, 0], "logvar2": logvar[:, 1], "eps1": eps[:, 0], "eps2": eps[:, 1]})
    dec = pd.DataFrame({"id": np.arange(1, n + 1)})
    for j in range(4):
        dec[f"x{j + 1}"] = x[:, j]
    for j in range(4):
        dec[f"xhat{j + 1}"] = x_hat[:, j]
    return {"input.csv": enc, "decoder.csv": dec}


def data_gan(rng):
    rows = []
    real_level = [0.95, 0.9, 0.8, 0.7, 0.6]                        # D всё менее уверен в реальных
    fake_level = [0.02, 0.08, 0.2, 0.35, 0.45]                     # генератор учится обманывать D
    for b in range(5):
        for kind, level in [("real", real_level[b]), ("fake", fake_level[b])]:
            vals = np.clip(level + rng.normal(0, 0.03 if level < 0.1 else 0.06, 4), 0.005, 0.995).round(3)
            for v in vals:
                rows.append({"batch": b + 1, "kind": kind, "d_out": v})
    return pd.DataFrame(rows)


def data_object_detection(rng):
    boxes = [
        ("cat", 10, 10, 60, 60, 0.92), ("cat", 12, 8, 62, 58, 0.85), ("cat", 5, 15, 55, 65, 0.70),
        ("cat", 8, 12, 58, 60, 0.40), ("cat", 30, 30, 80, 80, 0.60), ("cat", 30, 10, 80, 60, 0.50),
        ("cat", 100, 20, 150, 90, 0.88), ("cat", 104, 24, 152, 92, 0.55),
        ("dog", 40, 40, 100, 100, 0.81), ("dog", 42, 38, 98, 104, 0.77), ("dog", 12, 10, 60, 62, 0.66),
        ("dog", 45, 50, 105, 110, 0.45),
    ]
    order = rng.permutation(len(boxes))                            # id не совпадают с порядком по score
    df = pd.DataFrame([boxes[i] for i in order], columns=["cls", "x1", "y1", "x2", "y2", "score"])
    df.insert(0, "box_id", np.arange(1, len(df) + 1))
    return df


def data_segmentation(rng):
    rows = []
    size = 5
    masks = {}
    t = np.zeros((size, size), int); t[1:4, 1:4] = 1
    p = np.zeros((size, size), int); p[1:4, 2:5] = 1
    masks["img1"] = (t, p)                                         # предсказание сдвинуто на пиксель
    t = np.zeros((size, size), int); t[2, 2] = 1; t[2, 3] = 1
    p = np.zeros((size, size), int); p[1:4, 1:4] = 1
    masks["img2"] = (t, p)                                         # маленький объект, «раздутое» предсказание
    masks["img3"] = (np.zeros((size, size), int), np.zeros((size, size), int))   # объекта нет и не найден
    t = np.zeros((size, size), int)
    p = np.zeros((size, size), int); p[0, 0] = 1; p[4, 3] = 1
    masks["img4"] = (t, p)                                         # ложные срабатывания на пустом снимке
    t = (rng.uniform(size=(size, size)) < 0.5).astype(int)
    p = t.copy(); flip = rng.choice(size * size, 5, replace=False); p.flat[flip] = 1 - p.flat[flip]
    masks["img5"] = (t, p)                                         # «шумная» маска с 5 ошибками
    for img, (t, p) in masks.items():
        for r in range(size):
            for c in range(size):
                rows.append({"image_id": img, "row": r, "col": c, "true": t[r, c], "pred": p[r, c]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    # ------------------------------------------------------------ Путь 1
    {
        "name": "gradient_descent",
        "title": "Градиентный спуск для линейной регрессии вручную",
        "data": data_gradient_descent,
        "task": """
            Дан файл `input.csv`: площадь `area`, возраст дома `age` и цена `price` (40 квартир).
            Обучите линейную регрессию price ≈ w₁·area′ + w₂·age′ + b **пакетным градиентным спуском**,
            написанным вручную (без sklearn для обучения).

            1. Стандартизируйте признаки по всем 40 строкам: x′ = (x − mean) / std, std — с `ddof=0`
               (как `StandardScaler`). Целевую `price` не масштабируйте.
            2. Старт: w₁ = w₂ = b = 0. Скорость обучения **α = 0.1**, **100 эпох**; эпоха = один шаг по всей выборке.
            3. Функция потерь — MSE: L = (1/n)·Σ(ŷᵢ − yᵢ)². Градиенты:
               ∂L/∂w = (2/n)·X′ᵀ(ŷ − y), ∂L/∂b = (2/n)·Σ(ŷᵢ − yᵢ).
            4. На каждой эпохе: прямой проход → loss → градиенты → шаг θ ← θ − α·∇L.
               **loss эпохи** — MSE, посчитанная в прямом проходе этой эпохи, т.е. **до** шага;
               веса в строке — **после** шага этой эпохи.

            ## Подсказки

            - Удобно держать признаки в матрице X′ размера 40 × 2, а ŷ считать как X′·w + b — без циклов по объектам.
            - Проверьте себя: loss на эпохе 1 равен mean(y²) (веса нулевые), а дальше он должен монотонно убывать.
            - Цикл по эпохам: `err = X @ w + b - y`, `loss = mean(err**2)`, `grad_w = 2 / n * X.T @ err`,
              `grad_b = 2 * mean(err)`, затем одновременно обновите w и b. В конце сравните w, b с решением
              `LinearRegression` на тех же стандартизованных признаках — они должны почти совпасть.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, loss, w1, w2, b` — 11 строк для эпох 1, 10, 20, …, 100 (по возрастанию);
            `w1` — вес `area`, `w2` — вес `age`. Значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LinearRegression

            LR, EPOCHS = 0.1, 100


            def main():
                df = pd.read_csv('input.csv')
                X = df[['area', 'age']].to_numpy(dtype=float)
                y = df['price'].to_numpy()
                n = len(y)

                # Стандартизация (ddof=0, как StandardScaler): без неё нужен крошечный lr
                X = (X - X.mean(axis=0)) / X.std(axis=0)

                w, b = np.zeros(2), 0.0
                rows = []
                for epoch in range(1, EPOCHS + 1):
                    err = X @ w + b - y                 # прямой проход: ŷ − y
                    loss = np.mean(err ** 2)            # MSE до шага
                    grad_w = 2 / n * X.T @ err          # ∂L/∂w
                    grad_b = 2 * np.mean(err)           # ∂L/∂b
                    w, b = w - LR * grad_w, b - LR * grad_b
                    if epoch == 1 or epoch % 10 == 0:
                        rows.append({'epoch': epoch, 'loss': loss, 'w1': w[0], 'w2': w[1], 'b': b})

                # Сверка с аналитическим решением (нормальное уравнение / LinearRegression)
                ols = LinearRegression().fit(X, y)
                print(f"GD:  w = {w.round(4)}, b = {b:.4f}")
                print(f"OLS: w = {ols.coef_.round(4)}, b = {ols.intercept_:.4f}")
                assert np.allclose(w, ols.coef_, atol=1e-3) and np.isclose(b, ols.intercept_, atol=1e-3)

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Градиентный спуск повторяет одно и то же: прямой проход ŷ = X′w + b, ошибка e = ŷ − y,
            градиент ∇L = (2/n)·X′ᵀe и шаг против градиента. На стандартизованных признаках поверхность MSE почти
            «круглая», поэтому α = 0.1 сходится за десятки эпох к тому же решению, что и нормальное уравнение.

            **Типичные ошибки:**
            - Забыт множитель 2 в градиенте — фактически вдвое меньший шаг: loss убывает медленнее, веса на эпохах 10–50
              не совпадают (к эпохе 100 почти догоняют).
            - Обновили b уже новыми w (или посчитали градиент b после шага по w) — траектория чуть другая.
            - Записали loss после шага, а не в прямом проходе эпохи — весь столбец `loss` сдвинут на одну эпоху.
            - Не стандартизировали признаки — при α = 0.1 спуск расходится (area ~ 100 даёт огромный градиент).

            **Что вынести:** GD = forward → loss → gradient → step; масштаб признаков определяет допустимый lr,
            а результат после сходимости совпадает с аналитическим. → шпаргалка «Градиентный спуск».
        """,
    },
    {
        "name": "knn",
        "title": "k ближайших соседей вручную",
        "data": data_knn,
        "task": """
            Дан файл `input.csv` с заявками: доход `income` (рубли), доля долга `debt_ratio` (0–1),
            метка `default` (1 — клиент не вернул кредит) и `split` (`train`/`test`).

            Реализуйте классификатор **k-NN** вручную:

            1. Масштабируйте `income, debt_ratio` с помощью `StandardScaler`, **обученного только на train**.
            2. Для каждой тестовой строки найдите **k = 5** ближайших train-объектов по **евклидову** расстоянию
               в масштабированном пространстве.
            3. `proba` — доля соседей с `default = 1`; `pred` = 1, если proba > 0.5, иначе 0
               (k нечётное и классов два, поэтому ничьих при голосовании не бывает; совпадающих расстояний
               на границе пятёрки в данных тоже нет).
            4. Сверьте результат с `KNeighborsClassifier(n_neighbors=5)`.

            ## Подсказки

            - Посмотрите на масштабы: доход — десятки тысяч, доля долга — от 0 до 1. Что будет решать расстояние без масштабирования?
            - Матрицу всех расстояний test × train можно получить broadcasting'ом: `X_test[:, None, :] - X_train[None, :, :]`.
            - Для каждой тестовой строки возьмите `np.argsort(dist)[:5]`, посмотрите метки этих соседей и усредните их —
              это `proba`; сравните с `predict_proba(X_test)[:, 1]` из sklearn.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, proba, pred` для тестовых строк в исходном порядке; `proba` округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['income', 'debt_ratio']
            K = 5


            def knn_proba(X_train, y_train, X_test, k=K):
                # Евклидовы расстояния всех test до всех train: матрица n_test × n_train
                dist = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
                order = np.argsort(dist, axis=1)
                # Проверка условия: k-й и (k+1)-й соседи не на одинаковом расстоянии
                d_sorted = np.take_along_axis(dist, order, axis=1)
                assert np.all(d_sorted[:, k] - d_sorted[:, k - 1] > 1e-9)
                return y_train[order[:, :k]].mean(axis=1)


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                y_train = train['default'].to_numpy()

                scaler = StandardScaler().fit(train[FEATURES])       # только train!
                X_train = scaler.transform(train[FEATURES])
                X_test = scaler.transform(test[FEATURES])

                proba = knn_proba(X_train, y_train, X_test)
                pred = (proba > 0.5).astype(int)

                # Сверка с библиотекой
                clf = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
                assert np.allclose(proba, clf.predict_proba(X_test)[:, 1])
                assert np.array_equal(pred, clf.predict(X_test))

                # Для сравнения: без масштабирования соседей выбирает почти только доход
                raw = knn_proba(train[FEATURES].to_numpy(float), y_train, test[FEATURES].to_numpy(float))
                acc = lambda p: np.mean((p > 0.5) == test['default'].to_numpy())
                print(f"accuracy со StandardScaler: {acc(proba):.3f}   без масштабирования: {acc(raw):.3f}")
                print("предсказаний изменилось:", int(((raw > 0.5) != (proba > 0.5)).sum()))

                return pd.DataFrame({'id': test['id'], 'proba': proba.round(4), 'pred': pred})
        """,
        "review": """
            **Идея.** У k-NN нет обучения: предсказание — голосование k ближайших train-объектов. Поэтому всё решает
            метрика расстояния, а она чувствительна к масштабу: без StandardScaler разница в доходе в 1000 ₽ «весит»
            в тысячу раз больше, чем разница в доле долга от 0 до 1.

            **Типичные ошибки:**
            - Не масштабировали признаки — соседи подбираются по доходу, `debt_ratio` почти не влияет, часть `pred` меняется.
            - Обучили scaler на всех данных (или отдельно на test) — среднее и std другие, расстояния и иногда соседи меняются.
            - Взяли `argsort(...)[:k]` по строкам не той оси или включили test-объекты в число соседей.
            - Порог `proba >= 0.5` вместо `> 0.5` здесь не опасен (k = 5), но при чётном k именно он решает ничьи.

            **Что вынести:** для методов на расстояниях (k-NN, k-means, SVM с RBF) масштабирование обязательно,
            а scaler обучается только на train. → шпаргалка «k-NN».
        """,
    },
    {
        "name": "naive_bayes",
        "title": "Мультиномиальный наивный Байес для спама",
        "data": data_naive_bayes,
        "task": """
            Дан файл `input.csv` с короткими сообщениями: `text` (слова через пробел), метка `label`
            (`ham` или `spam`) и `split` (20 строк `train`, 6 строк `test`).

            Реализуйте **мультиномиальный наивный Байес** вручную:

            1. Словарь V — все различные слова train-сообщений (разбиение по пробелам). Слова test-сообщений,
               которых нет в V, игнорируются.
            2. Априорные вероятности: P(c) = (число train-сообщений класса c) / 20.
            3. Правдоподобие слова со **сглаживанием Лапласа α = 1**:
               P(w | c) = (N_wc + 1) / (N_c + |V|), где N_wc — сколько раз слово w встретилось во всех train-сообщениях
               класса c, N_c — общее число слов (с повторами) в train-сообщениях класса c.
            4. Для каждого test-сообщения — совместная логарифмическая вероятность
               log P(c) + Σ_w n_w · log P(w | c), где n_w — сколько раз w встречается в сообщении.
               Логарифм натуральный.
            5. `p_spam` — апостериорная вероятность спама (нормируйте две экспоненты), `pred` — класс с большей
               логарифмической вероятностью.

            Сверьте результат с `CountVectorizer(token_pattern=r"\\S+")` + `MultinomialNB(alpha=1.0)`.

            ## Подсказки

            - Перемножение многих маленьких вероятностей быстро уходит в ноль — поэтому складывают логарифмы.
            - Без сглаживания одно незнакомое для класса слово обнулит всю вероятность класса. Именно для этого +1.
            - Постройте матрицу счётчиков «сообщение × слово» (например, через `Counter`), просуммируйте строки
              по классам → N_wc, затем `log((N_wc + 1) / (N_wc.sum() + |V|))`; совместная вероятность —
              `log_prior + counts_test @ log_lik`, у sklearn это `predict_joint_log_proba`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, logp_ham, logp_spam, p_spam, pred` для test-строк в исходном порядке;
            числа округлить до 4 знаков, `pred` — строка `ham` или `spam`.
        """,
        "solution": """
            from collections import Counter

            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.naive_bayes import MultinomialNB

            CLASSES = ['ham', 'spam']
            ALPHA = 1.0


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                vocab = sorted({w for text in train['text'] for w in text.split()})
                index = {w: j for j, w in enumerate(vocab)}

                def counts(texts):
                    M = np.zeros((len(texts), len(vocab)))
                    for i, text in enumerate(texts):
                        for w, c in Counter(text.split()).items():
                            if w in index:                       # незнакомые слова игнорируем
                                M[i, index[w]] = c
                    return M

                X_train, X_test = counts(list(train['text'])), counts(list(test['text']))
                y = train['label'].to_numpy()

                log_prior = np.log([np.mean(y == c) for c in CLASSES])
                N = np.array([X_train[y == c].sum(axis=0) for c in CLASSES])          # N_wc: 2 × |V|
                log_lik = np.log((N + ALPHA) / (N.sum(axis=1, keepdims=True) + ALPHA * len(vocab)))

                joint = log_prior + X_test @ log_lik.T                                 # n_test × 2
                p_spam = 1 / (1 + np.exp(joint[:, 0] - joint[:, 1]))                   # нормировка двух экспонент
                pred = np.array(CLASSES)[joint.argmax(axis=1)]

                # Сверка со sklearn
                vec = CountVectorizer(token_pattern=r'\\S+', lowercase=False)
                nb = MultinomialNB(alpha=ALPHA).fit(vec.fit_transform(train['text']), y)
                Xt = vec.transform(test['text'])
                assert list(nb.classes_) == CLASSES
                assert np.allclose(joint, nb.predict_joint_log_proba(Xt))
                assert np.allclose(p_spam, nb.predict_proba(Xt)[:, 1])
                print(f"|V| = {len(vocab)}; самые «спамные» слова:",
                      [vocab[j] for j in np.argsort(log_lik[1] - log_lik[0])[::-1][:3]])

                return pd.DataFrame({'id': test['id'], 'logp_ham': joint[:, 0], 'logp_spam': joint[:, 1],
                                     'p_spam': p_spam, 'pred': pred}).round(4)
        """,
        "review": """
            **Идея.** Наивный Байес считает слова независимыми при известном классе, поэтому
            log P(c | текст) = log P(c) + Σ n_w·log P(w | c) + const. Обучение — это просто подсчёт слов по классам,
            а сглаживание Лапласа не даёт нулевых вероятностей для слов, не встречавшихся в классе.

            **Типичные ошибки:**
            - В знаменателе N_c + 1 вместо N_c + |V| — вероятности слов не суммируются в 1, все логарифмы смещены.
            - Учли слово «кот», которого нет в train (добавили его в словарь) — |V| на 1 больше, все правдоподобия чуть другие.
            - Считали наличие слова (0/1), а не число вхождений n_w — это уже Бернуллиевский NB, ответ другой.
            - Забыли априорную вероятность log P(c) — при несбалансированных классах меняются и logp, и иногда `pred`.

            **Что вынести:** NB — быстрый сильный бейзлайн для текстов; работаем в логарифмах и всегда сглаживаем.
            → шпаргалка «Naive Bayes».
        """,
    },
    {
        "name": "roc_auc",
        "title": "ROC-кривая и AUC вручную",
        "data": data_roc_auc,
        "task": """
            Дан файл `input.csv`: истинный класс `y_true` и оценка модели `y_score` для 20 объектов.
            Оценки округлены до 0.1, поэтому **есть одинаковые значения** — в том числе у объектов разных классов.

            1. Возьмите все **уникальные** значения `y_score` и отсортируйте **по убыванию** — это пороги t.
            2. Для каждого порога предсказываем класс 1, если `y_score >= t`. Посчитайте TP, FP,
               TPR = TP / P и FPR = FP / N (P и N — число объектов классов 1 и 0).
            3. Площадь под кривой — **методом трапеций**, начиная с точки (FPR, TPR) = (0, 0):
               `auc_cum` в строке — площадь от (0, 0) до текущей точки включительно,
               т.е. auc_cum_k = auc_cum_(k−1) + (FPR_k − FPR_(k−1)) · (TPR_k + TPR_(k−1)) / 2.
               В последней строке (порог = минимальная оценка) TPR = FPR = 1, а `auc_cum` — это ROC-AUC.

            Сверьте точки с `roc_curve(y_true, y_score, drop_intermediate=False)` (у sklearn есть ещё первая точка
            с порогом `inf` и TPR = FPR = 0 — в ответ её **не включайте**) и AUC с `roc_auc_score`.

            ## Подсказки

            - Порог «≥ t» означает: при снижении порога в класс 1 переходят все объекты с оценкой t сразу — и позитивные, и негативные.
            - Если при одном пороге добавились и позитивы, и негативы, кривая идёт **по диагонали** — трапеция,
              а не «ступенька». Именно так AUC учитывает совпадающие оценки (как половину пары).
            - Цикл по `np.sort(np.unique(score))[::-1]`: `tp = ((score >= t) & (y == 1)).sum()`, аналогично `fp`;
              площадь накапливайте `np.cumsum` по `np.diff` с добавленной точкой (0, 0).

            ## Ожидаемый формат выходного файла

            CSV с колонками `threshold, tp, fp, tpr, fpr, auc_cum` — по строке на уникальный порог,
            пороги по убыванию; дробные значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import roc_auc_score, roc_curve


            def main():
                df = pd.read_csv('input.csv')
                y, score = df['y_true'].to_numpy(), df['y_score'].to_numpy()
                P, N = (y == 1).sum(), (y == 0).sum()

                thresholds = np.sort(np.unique(score))[::-1]
                tp = np.array([((score >= t) & (y == 1)).sum() for t in thresholds])
                fp = np.array([((score >= t) & (y == 0)).sum() for t in thresholds])
                tpr, fpr = tp / P, fp / N

                # Трапеции от точки (0, 0): при совпадающих оценках сегмент идёт по диагонали
                fpr0, tpr0 = np.r_[0, fpr], np.r_[0, tpr]
                auc_cum = np.cumsum(np.diff(fpr0) * (tpr0[1:] + tpr0[:-1]) / 2)

                # Сверка со sklearn: первая точка (inf, 0, 0) — служебная
                fpr_sk, tpr_sk, thr_sk = roc_curve(y, score, drop_intermediate=False)
                assert np.isinf(thr_sk[0])
                assert np.allclose(thr_sk[1:], thresholds) and np.allclose(tpr_sk[1:], tpr) and np.allclose(fpr_sk[1:], fpr)
                assert np.isclose(auc_cum[-1], roc_auc_score(y, score))
                print(f"P={P}, N={N}, ROC-AUC = {auc_cum[-1]:.4f}")

                return pd.DataFrame({'threshold': thresholds, 'tp': tp, 'fp': fp, 'tpr': tpr, 'fpr': fpr,
                                     'auc_cum': auc_cum}).round(4)
        """,
        "review": """
            **Идея.** ROC-кривая — это TPR и FPR при всех возможных порогах. Пороги достаточно брать из самих оценок:
            между соседними уникальными значениями предсказания не меняются. AUC — площадь под ломаной, равная
            вероятности того, что случайный позитив получит оценку выше случайного негатива (ничья — ½).

            **Типичные ошибки:**
            - Сортировка объектов и пошаговое добавление по одному вместо группировки по уникальным порогам —
              строк больше, а при совпадающих оценках получаются «ступеньки» и AUC зависит от порядка строк.
            - Площадь прямоугольниками (`Δfpr · tpr`) вместо трапеций — AUC занижен ровно на вклад диагональных отрезков.
            - Порог `>` вместо `>=` — все точки сдвигаются на один порог, последняя строка не доходит до (1, 1).
            - Пороги по возрастанию — кривая идёт из (1, 1) в (0, 0), `auc_cum` накапливается с другого конца.

            **Что вынести:** ROC-AUC не зависит от выбора порога и от монотонных преобразований оценок;
            совпадающие оценки дают диагональные сегменты. → шпаргалка «ROC-AUC».
        """,
    },
    {
        "name": "random_forest",
        "title": "Случайный лес: OOB-оценка и важность признаков",
        "data": data_random_forest,
        "task": """
            Дан файл `input.csv` с данными об оттоке клиентов: `tenure` (месяцев с компанией), `monthly_charge`,
            `support_calls`, `noise` (случайный шум — не связан с целевой), `churn` (целевая) и `split`.

            1. Обучите `RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=42)`
               на train, признаки: `tenure, monthly_charge, support_calls, noise` (в этом порядке).
            2. Получите **OOB-оценку** (`oob_score_`) и **accuracy на test**.
            3. Получите важности признаков `feature_importances_` (MDI).

            ## Подсказки

            - Каждое дерево обучается на бутстрэп-выборке; примерно треть train-строк в неё не попадает — это «out-of-bag».
            - OOB-оценка — accuracy, где каждую train-строку предсказывают только деревья, которые её не видели.
              Сравните её с accuracy на test: обе — честные оценки, но OOB не тратит данные на отложенную выборку.
            - Всё нужное — атрибуты обученной модели: `rf.oob_score_`, `rf.feature_importances_`, а accuracy на test —
              `accuracy_score(y_test, rf.predict(X_test))`. Какая важность у признака `noise` и почему не нулевая?

            ## Ожидаемый формат выходного файла

            CSV в «длинном» формате с колонками `metric, value` и строками в порядке:
            `oob_score`, `test_accuracy`, `importance_tenure`, `importance_monthly_charge`,
            `importance_support_calls`, `importance_noise`; `value` округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            FEATURES = ['tenure', 'monthly_charge', 'support_calls', 'noise']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=42)
                rf.fit(train[FEATURES], train['churn'])

                # OOB: для каждой train-строки голосуют только деревья, в чью бутстрэп-выборку она не попала
                oob_pred = rf.oob_decision_function_.argmax(axis=1)
                assert np.isclose(rf.oob_score_, np.mean(rf.classes_[oob_pred] == train['churn'].to_numpy()))

                test_acc = accuracy_score(test['churn'], rf.predict(test[FEATURES]))
                assert np.isclose(rf.feature_importances_.sum(), 1)
                print(f"OOB = {rf.oob_score_:.3f}, test accuracy = {test_acc:.3f}")

                rows = [('oob_score', rf.oob_score_), ('test_accuracy', test_acc)]
                rows += [(f'importance_{f}', imp) for f, imp in zip(FEATURES, rf.feature_importances_)]
                return pd.DataFrame(rows, columns=['metric', 'value']).round(4)
        """,
        "review": """
            **Идея.** Случайный лес усредняет много глубоких некоррелированных деревьев: каждое учится на бутстрэп-выборке
            и на случайных подмножествах признаков. Бутстрэп даёт бесплатную валидацию — OOB: около 37 % строк
            ((1 − 1/n)ⁿ ≈ e⁻¹) каждое дерево не видело, и на них оно делает «честные» предсказания.

            **Типичные ошибки:**
            - Не указали `random_state` или изменили порядок признаков — другие бутстрэп-выборки и разбиения, другие числа.
            - Посчитали accuracy на train вместо OOB — получится сильно завышенная оценка (лес почти запоминает train).
            - Обучили лес на всех строках, включая test, — test accuracy завышена, OOB тоже изменится.
            - Прочитали ненулевую важность `noise` как полезность: MDI завышает важность непрерывных признаков
              с множеством возможных порогов — проверяйте permutation importance на отложенной выборке.

            **Что вынести:** OOB-оценка ≈ кросс-валидация «бесплатно»; важность MDI — быстрая, но смещённая.
            → шпаргалка «Random Forest».
        """,
    },
    # ------------------------------------------------------------ Путь 2
    {
        "name": "pytorch_full_guide",
        "title": "Цикл обучения как в PyTorch — на NumPy",
        "data": data_training_loop,
        "task": """
            Напишите на NumPy полный цикл обучения нейросети — ровно то, что делает типичный код на PyTorch:

            ```python
            model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            loss_fn = nn.MSELoss()
            for epoch in range(30):
                for xb, yb in batches:               # батчи 1, 2, 3 — без перемешивания
                    optimizer.zero_grad()            # обнулить градиенты
                    loss = loss_fn(model(xb), yb)    # forward + loss
                    loss.backward()                  # backward: посчитать градиенты
                    optimizer.step()                 # шаг SGD с моментом
            ```

            Данные — `input.csv`: признаки `x1, x2`, цель `y` и номер батча `batch` (по 8 строк в батче).
            Начальные веса — `weights.json` в раскладке PyTorch: `W1` (3 × 2), `b1` (3), `W2` (1 × 3), `b2` (1).

            1. **Forward**: h = tanh(xb·W1ᵀ + b1), ŷ = h·W2ᵀ + b2.
            2. **Loss**: MSE по батчу, L = mean((ŷ − y)²).
            3. **Backward**: градиенты L по W1, b1, W2, b2 (цепное правило; tanh′ = 1 − tanh²).
            4. **Step** (SGD с моментом μ = 0.9, lr = 0.1, как в PyTorch): для каждого параметра p с градиентом g
               v ← μ·v + g, p ← p − lr·v; в начале v = 0 (буфер момента **не** сбрасывается между батчами и эпохами).
            5. Порядок в каждой эпохе: батч 1, батч 2, батч 3; строки внутри батча — в порядке файла.

            **loss эпохи** — среднее трёх батчевых loss, посчитанных в forward этой эпохи (до соответствующих шагов) —
            как `running_loss / len(loader)` в PyTorch.

            ## Подсказки

            - Сначала напишите и проверьте forward и loss для первого батча при начальных весах, потом — backward.
            - Проверьте градиенты численно: (L(p + ε) − L(p − ε)) / 2ε для пары элементов W1 — должно совпасть.
            - Обратный проход: dŷ = 2(ŷ − y)/m, dW2 = dŷᵀ·h, db2 = Σdŷ, dh = dŷ·W2, dz = dh·(1 − h²),
              dW1 = dzᵀ·xb, db1 = Σdz. «zero_grad» в NumPy — это просто считать градиенты заново на каждом батче,
              а не прибавлять к старым.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, loss` — 30 строк для эпох 1…30; `loss` округлить до 4 знаков.
        """,
        "solution": """
            import json

            LR, MOMENTUM, EPOCHS = 0.1, 0.9, 30


            def forward(p, X):
                h = np.tanh(X @ p['W1'].T + p['b1'])             # nn.Linear(2, 3) + nn.Tanh()
                return h, h @ p['W2'].T + p['b2']                # nn.Linear(3, 1)


            def loss_fn(p, X, y):
                return np.mean((forward(p, X)[1] - y) ** 2)      # nn.MSELoss()


            def backward(p, X, y):
                # loss.backward(): цепное правило от L к каждому параметру
                h, y_hat = forward(p, X)
                d_yhat = 2 * (y_hat - y) / len(y)                # ∂L/∂ŷ  (m × 1)
                d_h = d_yhat @ p['W2']                           # ∂L/∂h  (m × 3)
                d_z = d_h * (1 - h ** 2)                         # через tanh
                return {'W2': d_yhat.T @ h, 'b2': d_yhat.sum(axis=0),
                        'W1': d_z.T @ X, 'b1': d_z.sum(axis=0)}


            def main():
                df = pd.read_csv('input.csv')
                params = {k: np.array(v, dtype=float) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                batches = [(g[['x1', 'x2']].to_numpy(), g[['y']].to_numpy()) for _, g in df.groupby('batch', sort=True)]

                # Численная проверка градиента на первом батче (до обучения)
                X0, y0 = batches[0]
                grads = backward(params, X0, y0)
                eps = 1e-6
                for (i, j) in [(0, 0), (2, 1)]:
                    plus = {k: v.copy() for k, v in params.items()}; plus['W1'][i, j] += eps
                    minus = {k: v.copy() for k, v in params.items()}; minus['W1'][i, j] -= eps
                    numeric = (loss_fn(plus, X0, y0) - loss_fn(minus, X0, y0)) / (2 * eps)
                    assert np.isclose(numeric, grads['W1'][i, j], atol=1e-6)

                velocity = {k: np.zeros_like(v) for k, v in params.items()}  # буферы момента
                rows = []
                for epoch in range(1, EPOCHS + 1):
                    running = 0.0
                    for X, y in batches:
                        running += loss_fn(params, X, y)          # forward + loss (до шага)
                        grads = backward(params, X, y)            # zero_grad + backward: градиенты «с нуля»
                        for k in params:                          # optimizer.step()
                            velocity[k] = MOMENTUM * velocity[k] + grads[k]
                            params[k] -= LR * velocity[k]
                    rows.append({'epoch': epoch, 'loss': running / len(batches)})

                print(f"loss: эпоха 1 = {rows[0]['loss']:.4f} → эпоха {EPOCHS} = {rows[-1]['loss']:.4f}")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Любой цикл обучения в PyTorch — четыре строки: `zero_grad()` → `loss = loss_fn(model(x), y)` →
            `loss.backward()` → `optimizer.step()`. Autograd делает за вас ровно то, что вы написали в `backward`:
            цепное правило от loss к каждому параметру. SGD с моментом копит «скорость» v и шагает по ней.

            ```python
            optimizer.zero_grad()      # grads = 0         (в NumPy: считаем градиенты заново)
            loss = loss_fn(model(xb), yb)
            loss.backward()            # grads = ∂L/∂p     (наша функция backward)
            optimizer.step()           # v = μv + g; p -= lr·v
            ```

            **Типичные ошибки:**
            - «Забыли zero_grad»: прибавляли новые градиенты к старым — фактически шаг растёт с каждым батчем, loss скачет или расходится.
            - Сбрасывали буфер момента v в начале каждой эпохи — с эпохи 2 траектория другая.
            - Использовали sum вместо mean в MSE — градиенты в 8 раз больше, при lr = 0.1 обучение расходится.
            - Перемножили на `W1` вместо `W1ᵀ` (перепутали раскладку `nn.Linear.weight` — out × in) — ошибка размерностей
              или, хуже, неверная сеть при квадратных матрицах.

            **Что вынести:** forward → loss → backward → step; градиенты не накапливаются между шагами,
            а состояние оптимизатора (момент) — накапливается. → шпаргалка «PyTorch: полное руководство».
        """,
    },
    {
        "name": "lstm",
        "title": "Ячейка LSTM: прямой проход по последовательности",
        "data": data_lstm,
        "task": """
            Дана последовательность из 5 шагов `input.csv` (`t, x1, x2`) и веса однослойного LSTM
            (вход D = 2, скрытое состояние H = 3) в `weights.json` в раскладке PyTorch `nn.LSTM`:
            `W_ih` (12 × 2), `W_hh` (12 × 3), `b_ih` (12), `b_hh` (12).

            Строки весов разбиты на 4 блока по H = 3 строки в порядке **i, f, g, o** (как в PyTorch):
            строки 0–2 — входной вентиль i, 3–5 — вентиль забывания f, 6–8 — кандидат g, 9–11 — выходной вентиль o.

            Для t = 1…5, начиная с h₀ = c₀ = 0:

            - a = W_ih·xₜ + b_ih + W_hh·hₜ₋₁ + b_hh (вектор из 12 чисел), a = [a_i, a_f, a_g, a_o]
            - iₜ = σ(a_i), fₜ = σ(a_f), gₜ = tanh(a_g), oₜ = σ(a_o)
            - cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ
            - hₜ = oₜ ⊙ tanh(cₜ)

            (⊙ — поэлементное умножение, σ — сигмоида.)

            ## Подсказки

            - Все четыре вентиля считаются одним матричным умножением — потом вектор режется на 4 части по 3.
            - Проверьте себя на шаге 1: h₀ = 0, поэтому вклад W_hh нулевой, но b_hh всё равно прибавляется.
            - Цикл по t: `a = W_ih @ x + b_ih + W_hh @ h + b_hh`, `i, f, g, o = np.split(a, 4)`, затем
              сигмоида для i, f, o и tanh для g; c и h обновляются по формулам выше.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, h1, h2, h3, c1, c2, c3` — 5 строк (t = 1…5); значения округлить до 4 знаков.
        """,
        "solution": """
            import json


            def sigmoid(x):
                return 1 / (1 + np.exp(-x))


            def main():
                df = pd.read_csv('input.csv')
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                H = w['W_hh'].shape[1]

                h, c = np.zeros(H), np.zeros(H)
                rows = []
                for t, x in zip(df['t'], df[['x1', 'x2']].to_numpy()):
                    a = w['W_ih'] @ x + w['b_ih'] + w['W_hh'] @ h + w['b_hh']   # все вентили сразу
                    a_i, a_f, a_g, a_o = np.split(a, 4)                          # порядок PyTorch: i, f, g, o
                    i, f, g, o = sigmoid(a_i), sigmoid(a_f), np.tanh(a_g), sigmoid(a_o)
                    c = f * c + i * g            # память: часть старой + часть нового
                    h = o * np.tanh(c)           # выход: отфильтрованная память
                    rows.append({'t': t, **{f'h{k + 1}': h[k] for k in range(H)},
                                 **{f'c{k + 1}': c[k] for k in range(H)}})
                    print(f"t={t}: f = {f.round(3)} (что сохранить), i = {i.round(3)} (что записать)")

                # h всегда в (−1, 1): o ∈ (0, 1), tanh(c) ∈ (−1, 1); c ничем не ограничена
                result = pd.DataFrame(rows)
                assert (result.filter(like='h').abs() < 1).all().all()
                return result.round(4)
        """,
        "review": """
            **Идея.** LSTM хранит две величины: «память» c и выход h. Вентиль забывания f решает, какую часть старой памяти
            оставить, входной i — сколько записать нового кандидата g, выходной o — что из памяти показать наружу.
            Аддитивное обновление cₜ = f⊙cₜ₋₁ + i⊙g позволяет градиенту проходить много шагов без затухания.

            **Типичные ошибки:**
            - Перепутали порядок блоков (например, i, f, o, g или Keras-порядок i, f, c, o) — все значения неверные с шага 1.
            - Забыли `b_hh` (в PyTorch два вектора смещений, и оба прибавляются) — ошибка уже на t = 1.
            - Применили сигмоиду к g вместо tanh — кандидат только положительный, c систематически растёт.
            - Записали в h значение tanh(c) без умножения на o или o⊙c без tanh.

            **Что вынести:** 4 вентиля = одно матричное умножение размера 4H; c — долгая память (аддитивная),
            h — короткая (отфильтрованная). → шпаргалка «LSTM».
        """,
    },
    {
        "name": "word_embeddings",
        "title": "Word embeddings: косинусная близость и аналогии",
        "data": data_word_embeddings,
        "task": """
            Файл `input.csv` — таблица эмбеддингов: 18 слов × 9 измерений (`word, e0…e8`).
            Файл `queries.csv` — запросы: `query_id, kind, a, b, c`.

            Косинусная близость: cos(u, v) = u·v / (‖u‖·‖v‖).

            - `kind = neighbors`: найти **3 ближайших** к слову `a` слова по косинусу, исключая само `a`.
            - `kind = analogy`: построить вектор v = e(a) − e(b) + e(c) (например, король − мужчина + женщина)
              и найти **3 ближайших** к v слова по косинусу, **исключая** слова a, b и c.

            Порядок: по убыванию косинуса; при равенстве (до 4-го знака) — по слову в алфавитном порядке.

            ## Подсказки

            - Нормируйте все векторы на длину один раз — тогда косинус со всеми словами = одно матричное умножение.
            - Без исключения слов запроса ближайшим к «король − мужчина + женщина» нередко оказывается сам «король».
            - Для запроса: `sims = E_norm @ (v / ‖v‖)`, выставьте `-inf` для исключённых слов и возьмите
              `np.argsort(-sims)[:3]` (или отсортируйте таблицу по паре (−cos, word)).

            ## Ожидаемый формат выходного файла

            CSV с колонками `query_id, rank, word, cosine` — по 3 строки на запрос (rank = 1, 2, 3), запросы в порядке
            `queries.csv`; `cosine` округлить до 4 знаков.
        """,
        "solution": """
            def main():
                table = pd.read_csv('input.csv')
                queries = pd.read_csv('queries.csv', keep_default_na=False)
                words = table['word'].tolist()
                E = table.drop(columns='word').to_numpy()
                E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
                vec = dict(zip(words, E))

                rows = []
                for _, q in queries.iterrows():
                    if q['kind'] == 'neighbors':
                        v, exclude = vec[q['a']], {q['a']}
                    else:
                        v = vec[q['a']] - vec[q['b']] + vec[q['c']]        # a − b + c
                        exclude = {q['a'], q['b'], q['c']}
                    sims = E_norm @ (v / np.linalg.norm(v))
                    cand = pd.DataFrame({'word': words, 'cosine': sims.round(4)})
                    cand = cand[~cand['word'].isin(exclude)]
                    # Тай-брейк: при равном (округлённом) косинусе — по алфавиту
                    top = cand.sort_values(['cosine', 'word'], ascending=[False, True]).head(3)
                    for rank, (_, r) in enumerate(top.iterrows(), start=1):
                        rows.append({'query_id': q['query_id'], 'rank': rank, 'word': r['word'], 'cosine': r['cosine']})
                    print(q['kind'], q['a'], q['b'], q['c'], '→', list(top['word']))

                # Проверка: косинус через sklearn
                from sklearn.metrics.pairwise import cosine_similarity
                assert np.allclose(cosine_similarity(E)[0], E_norm @ E_norm[0])
                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** В хорошем пространстве эмбеддингов смысл — это направление: близкие слова имеют большой косинус,
            а отношения («мужской → женский», «страна → столица») — примерно одинаковые векторы-сдвиги. Поэтому
            король − мужчина + женщина оказывается рядом с «королевой».

            **Типичные ошибки:**
            - Евклидово расстояние вместо косинуса — длина векторов начинает влиять, порядок соседей меняется.
            - Не исключили слова запроса — в аналогии первым часто выходит «король» (или «женщина»), все ранги сдвигаются.
            - Перепутали знаки: a + b − c вместо a − b + c — «аналогия» ищет совсем другое.
            - Сравнивали ненормированный v со строками E — для ранжирования это не страшно, но значения `cosine` неверны.

            **Что вынести:** косинус = скалярное произведение нормированных векторов; аналогии — арифметика сдвигов,
            и слова запроса всегда исключают. → шпаргалка «Word Embeddings».
        """,
    },
    {
        "name": "vae",
        "title": "VAE: репараметризация, KL и ELBO",
        "data": data_vae,
        "task": """
            Энкодер VAE для 6 объектов уже выдал параметры апостериора q(z | x) = N(μ, diag(σ²)) в латентном пространстве
            размерности 2: файл `input.csv` (`id, mu1, mu2, logvar1, logvar2, eps1, eps2`), где logvar = log σ²,
            а ε ~ N(0, I) — уже сэмплированный шум. Файл `decoder.csv` содержит исходные бинарные «картинки» из
            4 пикселей `x1…x4` и выход декодера `xhat1…xhat4` (вероятности пикселей).

            Для каждого объекта вычислите:

            1. **Репараметризация**: σ = exp(0.5·logvar), z = μ + σ·ε → `z1, z2`.
            2. **KL** к N(0, I): KL = −0.5 · Σⱼ (1 + logvarⱼ − μⱼ² − exp(logvarⱼ)) (сумма по 2 латентным координатам).
            3. **Реконструкция** — бинарная кросс-энтропия, **сумма** по 4 пикселям (как `F.binary_cross_entropy(..., reduction='sum')`):
               recon = −Σₖ [xₖ·log x̂ₖ + (1 − xₖ)·log(1 − x̂ₖ)].
            4. **ELBO** = −recon − KL (это то, что VAE **максимизирует**; loss = −ELBO).

            ## Подсказки

            - В файле logvar, а не σ: σ = √(exp(logvar)) = exp(logvar / 2). Зачем так хранят? logvar может быть любым числом.
            - Объект 1 — проверочный: μ = 0 и logvar = 0, т.е. q совпадает с априорным N(0, I), и KL должен быть ровно 0.
            - Векторно: `kl = -0.5 * (1 + logvar - mu**2 - np.exp(logvar)).sum(axis=1)`,
              `recon = -(x*np.log(xh) + (1-x)*np.log(1-xh)).sum(axis=1)`, а `elbo = -(recon + kl)` — всегда ≤ 0 для BCE.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, z1, z2, kl, recon, elbo` — 6 строк в порядке id; значения округлить до 4 знаков.
        """,
        "solution": """
            def main():
                enc = pd.read_csv('input.csv')
                dec = pd.read_csv('decoder.csv')
                mu = enc[['mu1', 'mu2']].to_numpy()
                logvar = enc[['logvar1', 'logvar2']].to_numpy()
                eps = enc[['eps1', 'eps2']].to_numpy()
                x = dec[[f'x{k}' for k in range(1, 5)]].to_numpy()
                x_hat = dec[[f'xhat{k}' for k in range(1, 5)]].to_numpy()

                # Репараметризация: случайность вынесена в ε, градиент проходит через μ и σ
                sigma = np.exp(0.5 * logvar)
                z = mu + sigma * eps

                # KL(N(μ, σ²) || N(0, 1)) в замкнутой форме, сумма по латентным координатам
                kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=1) + 0.0   # + 0.0: без «-0.0»
                # Численная сверка с общей формулой KL = log(1/σ) + (σ² + μ²)/2 − 1/2
                assert np.allclose(kl, np.sum(-np.log(sigma) + (sigma ** 2 + mu ** 2) / 2 - 0.5, axis=1))
                assert np.isclose(kl[0], 0)

                recon = -np.sum(x * np.log(x_hat) + (1 - x) * np.log(1 - x_hat), axis=1)   # BCE, сумма
                elbo = -(recon + kl)

                print("средний loss (−ELBO):", round(float(-elbo.mean()), 4), " средний KL:", round(float(kl.mean()), 4))
                return pd.DataFrame({'id': enc['id'], 'z1': z[:, 0], 'z2': z[:, 1],
                                     'kl': kl, 'recon': recon, 'elbo': elbo}).round(4)
        """,
        "review": """
            **Идея.** VAE учится максимизировать ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z)): декодер должен хорошо
            восстанавливать x, а апостериор не должен уходить далеко от N(0, I). Репараметризация z = μ + σ·ε выносит
            случайность в ε, поэтому через сэмплирование можно пропустить градиент к μ и σ.

            **Типичные ошибки:**
            - σ = exp(logvar) или σ = logvar вместо exp(0.5·logvar) — неверные z (а KL при этом может выйти верным).
            - Потеряли знак или 0.5 в KL — KL отрицательный или в 2 раза больше; KL ≥ 0 всегда, это простая самопроверка.
            - Усреднили BCE по пикселям (`reduction='mean'`) — recon в 4 раза меньше, баланс с KL нарушен.
            - ELBO = recon − KL или recon + KL — путаница между ELBO (максимизируем) и loss = −ELBO (минимизируем).

            **Что вынести:** loss VAE = реконструкция + KL; logvar → σ через exp(0.5·logvar); KL = 0 только при μ = 0, σ = 1.
            → шпаргалка «VAE».
        """,
    },
    {
        "name": "gan",
        "title": "Функции потерь GAN: minimax и non-saturating",
        "data": data_gan,
        "task": """
            Файл `input.csv` — выходы дискриминатора за 5 батчей обучения GAN: `batch`, `kind` (`real` — D(x) на реальных
            объектах, `fake` — D(G(z)) на сгенерированных) и `d_out` — вероятность «реальный» (по 4 объекта каждого вида в батче).

            Для каждого батча вычислите (log — натуральный, mean — по 4 объектам):

            1. **Loss дискриминатора** (BCE на реальных с меткой 1 + BCE на фейках с меткой 0, **сумма**, без деления на 2):
               d_loss = −mean(log D(x)) − mean(log(1 − D(G(z))))
            2. **Loss генератора в исходной minimax-формулировке**: g_loss_minimax = mean(log(1 − D(G(z))))
               (генератор его **минимизирует**).
            3. **Non-saturating loss генератора**: g_loss_ns = −mean(log D(G(z))).
            4. **Сила градиента** для генератора по логиту дискриминатора a (D = σ(a)) в среднем по фейкам:
               для minimax |∂/∂a log(1 − σ(a))| = D(G(z)) → `grad_minimax = mean(D(G(z)))`;
               для non-saturating |∂/∂a (−log σ(a))| = 1 − D(G(z)) → `grad_ns = mean(1 − D(G(z)))`.

            ## Подсказки

            - Разделите строки батча по `kind` и считайте всё через `groupby('batch')`.
            - Посмотрите на батч 1: дискриминатор легко отличает фейки (D(G(z)) ≈ 0.02). Какой из двух лоссов генератора
              при этом почти не меняется, а какой большой?
            - d_loss — это `BCE(D(x), 1) + BCE(D(G(z)), 0)`; g_loss_ns — это `BCE(D(G(z)), 1)`, то есть генератор учат
              с «перевёрнутой» меткой. Градиенты выведите через σ′(a) = σ(a)(1 − σ(a)).

            ## Ожидаемый формат выходного файла

            CSV с колонками `batch, d_loss, g_loss_minimax, g_loss_ns, grad_minimax, grad_ns` — 5 строк (batch = 1…5);
            значения округлить до 4 знаков.
        """,
        "solution": """
            def main():
                df = pd.read_csv('input.csv')
                rows = []
                for batch, g in df.groupby('batch', sort=True):
                    d_real = g.loc[g['kind'] == 'real', 'd_out'].to_numpy()
                    d_fake = g.loc[g['kind'] == 'fake', 'd_out'].to_numpy()

                    d_loss = -np.mean(np.log(d_real)) - np.mean(np.log(1 - d_fake))   # BCE(real, 1) + BCE(fake, 0)
                    g_minimax = np.mean(np.log(1 - d_fake))                          # минимизирует G в исходной игре
                    g_ns = -np.mean(np.log(d_fake))                                  # BCE(fake, 1)

                    # d/da log(1 − σ(a)) = −σ(a);  d/da (−log σ(a)) = −(1 − σ(a))
                    rows.append({'batch': batch, 'd_loss': d_loss, 'g_loss_minimax': g_minimax, 'g_loss_ns': g_ns,
                                 'grad_minimax': np.mean(d_fake), 'grad_ns': np.mean(1 - d_fake)})

                result = pd.DataFrame(rows)
                # Численная проверка производной на одном значении
                a, eps = 0.3, 1e-6
                s = lambda v: 1 / (1 + np.exp(-v))
                assert np.isclose((np.log(1 - s(a + eps)) - np.log(1 - s(a - eps))) / (2 * eps), -s(a), atol=1e-6)
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** В исходной игре генератор минимизирует log(1 − D(G(z))). Но в начале обучения D легко отличает фейки,
            D(G(z)) ≈ 0, и градиент этого лосса по логиту равен D(G(z)) ≈ 0 — генератор «насыщается» и почти не учится.
            Non-saturating вариант −log D(G(z)) имеет ту же неподвижную точку, но градиент 1 − D(G(z)) ≈ 1 как раз там,
            где генератор плох. Поэтому на практике используют именно его (BCE фейков с меткой 1).

            **Типичные ошибки:**
            - Поделили d_loss на 2 (так делают в некоторых туториалах) — ответ ровно вдвое меньше; в условии сказано «сумма».
            - Записали g_loss_minimax со знаком минус (−mean(log(1 − D))) — получилось положительное число, знак противоположный.
            - В non-saturating взяли log(1 − D(G(z))) вместо log D(G(z)) — это опять minimax.
            - Посчитали градиент по самому D, а не по логиту: 1/(1 − D) и 1/D — другая шкала, хотя вывод о насыщении тот же.

            **Что вынести:** D учится как обычный бинарный классификатор, G — с «перевёрнутой» меткой (non-saturating);
            следите, чтобы d_loss не падал к нулю — это сигнал, что генератор перестал получать градиент. → шпаргалка «GAN».
        """,
    },
    # ------------------------------------------------------------ CV
    {
        "name": "object_detection",
        "title": "IoU и Non-Maximum Suppression",
        "data": data_object_detection,
        "task": """
            Детектор выдал 12 рамок (`input.csv`): `box_id`, класс `cls`, координаты углов `x1, y1, x2, y2`
            (левый верхний и правый нижний, в пикселях) и уверенность `score`.

            **IoU** двух рамок = площадь пересечения / площадь объединения, площадь рамки = (x2 − x1)·(y2 − y1)
            (без «+1»); если рамки не пересекаются, пересечение = 0.

            Выполните **жадный NMS отдельно для каждого класса** с порогом **IoU > 0.5** (как `torchvision.ops.nms`):

            1. Отсортируйте рамки класса по `score` по убыванию (при равенстве — по `box_id` по возрастанию).
            2. Возьмите первую из оставшихся рамок — она остаётся; удалите все оставшиеся рамки **того же класса**,
               у которых IoU с ней **строго больше** 0.5.
            3. Повторяйте, пока рамки не закончатся.

            Для каждой оставленной рамки посчитайте `n_suppressed` — сколько рамок она удалила.

            ## Подсказки

            - Рамки разных классов друг друга не подавляют, даже если сильно перекрываются — проверьте на собаке и кошке
              в левом верхнем углу.
            - Пересечение: ширина = max(0, min(x2ᵃ, x2ᵇ) − max(x1ᵃ, x1ᵇ)), высота — аналогично по y.
            - Для каждого класса: `order = sorted(...)`; пока order не пуст — `best = order.pop(0)`,
              затем `order = [b for b in order if iou(best, b) <= 0.5]`, а разницу длин запишите в `n_suppressed`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `box_id, cls, score, n_suppressed` — только оставленные рамки, отсортированные по `score`
            по убыванию (все классы вместе; при равенстве — по `box_id`).
        """,
        "solution": """
            IOU_THR = 0.5


            def iou(a, b):
                # a, b — (x1, y1, x2, y2)
                w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = w * h
                area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
                return inter / (area(a) + area(b) - inter)


            def main():
                df = pd.read_csv('input.csv')
                boxes = {r.box_id: (r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()}

                # В данных нет IoU, «висящих» на пороге: ответ не зависит от > или >=
                ious = [iou(boxes[a], boxes[b]) for a in boxes for b in boxes if a < b]
                assert all(abs(v - IOU_THR) > 1e-6 for v in ious)

                kept = []
                for cls, g in df.groupby('cls'):
                    order = g.sort_values(['score', 'box_id'], ascending=[False, True])['box_id'].tolist()
                    while order:
                        best = order.pop(0)
                        rest = [b for b in order if iou(boxes[best], boxes[b]) <= IOU_THR]
                        kept.append({'box_id': best, 'n_suppressed': len(order) - len(rest)})
                        order = rest

                result = pd.DataFrame(kept).merge(df[['box_id', 'cls', 'score']], on='box_id')
                result = result.sort_values(['score', 'box_id'], ascending=[False, True])
                print(f"оставлено {len(result)} из {len(df)} рамок")
                return result[['box_id', 'cls', 'score', 'n_suppressed']].reset_index(drop=True)
        """,
        "review": """
            **Идея.** Детектор выдаёт много пересекающихся рамок на один объект. NMS оставляет самую уверенную и выбрасывает
            её «дубликаты» — рамки того же класса с большим IoU, затем повторяет для оставшихся. Рамки с умеренным
            перекрытием (IoU ≤ 0.5) считаются другими объектами и выживают.

            **Типичные ошибки:**
            - NMS без учёта класса — собака в левом верхнем углу подавляется кошкой (или наоборот), рамок в ответе меньше.
            - Подавляли все рамки сразу по IoU с самой первой, а не итеративно — рамки, удалённые не лидером, считаются не тем.
            - Площадь с «+1» ((x2 − x1 + 1)·…) — IoU немного выше, пограничные рамки могут исчезнуть.
            - Сортировка по возрастанию score — остаются худшие рамки кластера.

            **Что вынести:** IoU — основная мера перекрытия в детекции (и в mAP), NMS — жадный отбор по score
            по каждому классу; порог IoU регулирует, насколько близкие объекты считать разными. → шпаргалка «Object Detection».
        """,
    },
    {
        "name": "image_segmentation",
        "title": "Метрики сегментации: IoU, Dice, pixel accuracy",
        "data": data_segmentation,
        "task": """
            Файл `input.csv` — маски сегментации 5 изображений 5 × 5 в «длинном» формате:
            `image_id, row, col, true, pred` (1 — пиксель объекта, 0 — фон).

            Для каждого изображения вычислите (TP, FP, FN, TN — по пикселям класса 1):

            1. **IoU** (Jaccard) = TP / (TP + FP + FN)
            2. **Dice** = 2·TP / (2·TP + FP + FN)
            3. **pixel_acc** = (TP + TN) / (число пикселей)

            **Соглашение о пустых масках**: если на изображении объекта нет ни в разметке, ни в предсказании
            (TP + FP + FN = 0), то IoU = Dice = 1 (идеальный ответ «объекта нет»). Если объекта нет, а модель
            что-то нашла (или наоборот), формулы дают 0 — это корректно.

            Добавьте последнюю строку `mean` — **среднее по изображениям** (macro) каждой метрики.

            ## Подсказки

            - Переведите длинную таблицу в маски через `groupby('image_id')` — или считайте TP, FP, FN прямо по столбцам.
            - Посмотрите на img4: pixel accuracy 92 %, а IoU = 0. Почему pixel accuracy плохо подходит для мелких объектов?
            - TP = sum(true & pred), FP = sum(~true & pred), FN = sum(true & ~pred); при TP + FP + FN = 0
              верните 1. Dice = 2·IoU / (1 + IoU) — удобная проверка.

            ## Ожидаемый формат выходного файла

            CSV с колонками `image_id, iou, dice, pixel_acc` — строки img1…img5 в исходном порядке и строка `mean`
            в конце; значения округлить до 4 знаков.
        """,
        "solution": """
            def scores(t, p):
                tp = np.sum((t == 1) & (p == 1))
                fp = np.sum((t == 0) & (p == 1))
                fn = np.sum((t == 1) & (p == 0))
                tn = np.sum((t == 0) & (p == 0))
                if tp + fp + fn == 0:             # обе маски пустые — идеальный ответ
                    iou = dice = 1.0
                else:
                    iou = tp / (tp + fp + fn)
                    dice = 2 * tp / (2 * tp + fp + fn)
                return iou, dice, (tp + tn) / len(t)


            def main():
                df = pd.read_csv('input.csv')
                rows = []
                for img, g in df.groupby('image_id', sort=False):
                    iou, dice, acc = scores(g['true'].to_numpy(), g['pred'].to_numpy())
                    assert np.isclose(dice, 2 * iou / (1 + iou))           # Dice и IoU связаны монотонно
                    rows.append({'image_id': img, 'iou': iou, 'dice': dice, 'pixel_acc': acc})

                # Сверка со sklearn для непустых случаев
                from sklearn.metrics import f1_score, jaccard_score
                g = df[df['image_id'] == 'img1']
                assert np.isclose(jaccard_score(g['true'], g['pred']), rows[0]['iou'])
                assert np.isclose(f1_score(g['true'], g['pred']), rows[0]['dice'])   # Dice = F1 по пикселям

                result = pd.DataFrame(rows)
                mean = result[['iou', 'dice', 'pixel_acc']].mean()
                result = pd.concat([result, pd.DataFrame([{'image_id': 'mean', **mean.to_dict()}])], ignore_index=True)
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** IoU и Dice смотрят только на пиксели объекта (TP, FP, FN) и игнорируют огромный фон, поэтому подходят
            для сегментации, где объект занимает малую долю изображения. Dice = F1 по пикселям и всегда ≥ IoU:
            Dice = 2·IoU / (1 + IoU). Pixel accuracy высокая даже у модели, которая «рисует» мусор на пустом снимке.

            **Типичные ошибки:**
            - Не обработали пустые маски — на img3 деление 0/0 даёт NaN, и `mean` тоже становится NaN (или 0, если подставили 0).
            - Посчитали одну общую IoU по всем пикселям всех изображений (micro) вместо среднего по изображениям — `mean` другой.
            - Union как TP + FP + FN + TN или как |true| + |pred| без вычитания пересечения — IoU сильно занижен.
            - Dice = TP / (TP + FP + FN) (то есть та же IoU) — `dice` совпадает с `iou`.

            **Что вынести:** для сегментации основная метрика — IoU/Dice, усреднённые по изображениям (или по классам — mIoU);
            соглашение о пустых масках надо фиксировать явно. → шпаргалка «Image Segmentation».
        """,
    },
]


def main():
    print("Генерация упражнений для этапов учебных путей:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=501):
        build(exercise, seed)


if __name__ == "__main__":
    main()
