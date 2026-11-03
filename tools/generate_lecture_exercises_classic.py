#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по классическому ML, для которых не было практики.

Путь 1: смещение-дисперсия, переобучение/недообучение, полиномиальная регрессия, Ridge/Lasso/ElasticNet,
матрица ошибок, градиентный бустинг для классификации, XGBoost, LightGBM, CatBoost.
Путь 2: перцептрон.
Путь 3: Bayesian Optimization, Optuna/Hyperopt (TPE), важность признаков, частичные зависимости.

xgboost, lightgbm, catboost и optuna в среде курса не установлены, поэтому ключевая идея каждой
библиотеки реализуется вручную на NumPy (прирост G²/(H+λ), гистограммы, ordered target statistics, TPE).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_classic.py
"""

import numpy as np
import pandas as pd

from generate_basic_exercises import build, split_column

# ---------------------------------------------------------------- данные


def data_bias_variance(rng):
    f = lambda x: np.sin(2 * np.pi * x)
    rows = []
    for sample in range(1, 41):                                    # 40 независимых обучающих выборок
        x = rng.uniform(0, 1, 30).round(4)
        y = (f(x) + rng.normal(0, 0.3, 30)).round(4)
        rows.append(pd.DataFrame({"sample": sample, "x": x, "y": y}))
    grid = np.linspace(0, 1, 101).round(2)
    return {"input.csv": pd.concat(rows, ignore_index=True),
            "grid.csv": pd.DataFrame({"x": grid, "f": f(grid).round(6)})}


def data_overfitting_underfitting(rng):
    n_train, n_val = 400, 200
    x = rng.uniform(0, 1, n_train + n_val).round(6)               # 6 знаков — без совпадающих x
    y = np.sin(2 * np.pi * x) + 0.5 * x + rng.normal(0, 0.3, n_train + n_val)
    return pd.DataFrame({"id": np.arange(1, n_train + n_val + 1), "x": x, "y": y.round(6),
                         "split": ["train"] * n_train + ["val"] * n_val})


def data_polynomial_regression(rng):
    n = 80
    x1 = rng.uniform(-2, 2, n).round(3)
    x2 = rng.uniform(-2, 2, n).round(3)
    y = 2 + x1 - 2 * x2 + 1.5 * x1 * x2 + 0.8 * x2 ** 3 + rng.normal(0, 1, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1, "x2": x2, "y": y.round(3)})


def data_ridge_lasso_elasticnet(rng):
    n, p = 50, 20
    X = rng.normal(size=(n, p))
    X[:, 3] = X[:, 0] + rng.normal(0, 0.2, n)                     # x4 — почти копия x1
    y = 4 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + rng.normal(0, 2.0, n)
    df = pd.DataFrame(X.round(3), columns=[f"x{j}" for j in range(1, p + 1)])
    df.insert(0, "id", np.arange(1, n + 1))
    df["y"] = y.round(3)
    df["split"] = split_column(n, 0.4, rng)
    return df


def data_confusion_matrix(rng):
    # Сколько объектов каждого настоящего класса ушло в каждый прогноз (кот, пёс, лиса)
    counts = {"кот": [17, 5, 2], "пёс": [4, 15, 2], "лиса": [0, 3, 12]}
    labels = ["кот", "пёс", "лиса"]
    pairs = [(t, p) for t, row in counts.items() for p, k in zip(labels, row) for _ in range(k)]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    return pd.DataFrame({"id": np.arange(1, len(pairs) + 1),
                         "y_true": [t for t, _ in pairs], "y_pred": [p for _, p in pairs]})


def data_boosting(rng):
    n = 300
    x1 = rng.uniform(-3, 3, n).round(3)
    x2 = rng.uniform(-3, 3, n).round(3)
    logit = 2 * np.sin(x1) + 0.8 * x2 - 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1, "x2": x2, "y": y,
                         "split": split_column(n, 0.3, rng)})


def data_xgboost(rng):
    n = 40
    x1 = rng.uniform(0, 10, n).round(2)
    x2 = rng.integers(18, 71, n)
    logit = 1.5 * (x1 > 6) - 0.06 * (x2 - 40) - 0.5
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    F = (0.4 * (x1 > 5) - 0.2 + rng.normal(0, 0.3, n)).round(3)  # счёт уже построенных деревьев
    return pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1, "x2": x2, "y": y, "F": F})


def data_lightgbm(rng):
    n = 400
    x1 = rng.uniform(0, 10, n).round(3)
    x2 = rng.normal(50, 15, n).round(3)
    y = np.where(x1 > 6.3, 4.0, 0.0) + 0.04 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1, "x2": x2, "y": y.round(3)})


def data_catboost(rng):
    cities = ["Москва", "Казань", "Самара", "Томск"]
    rate = {"Москва": 0.3, "Казань": 0.7, "Самара": 0.45, "Томск": 0.5, "Сочи": 0.5}
    train_city = rng.choice(cities, 30, p=[0.45, 0.25, 0.23, 0.07])
    test_city = np.append(rng.choice(cities, 8, p=[0.4, 0.3, 0.2, 0.1]), ["Сочи", "Сочи"])
    city = np.concatenate([train_city, test_city])
    bought = np.array([rng.binomial(1, rate[c]) for c in city])
    return pd.DataFrame({"id": np.arange(1, 41), "city": city, "bought": bought,
                         "split": ["train"] * 30 + ["test"] * 10})


def data_perceptron(rng):
    rows = []
    while len(rows) < 40:
        x1, x2 = rng.uniform(-3, 3, 2).round(2)
        margin = 1.0 * x1 - 0.8 * x2 + 2.0
        if abs(margin) > 0.1:                                      # классы линейно разделимы с узким зазором
            rows.append({"id": len(rows) + 1, "x1": x1, "x2": x2, "y": int(margin > 0)})
    return pd.DataFrame(rows)


def data_feature_importance(rng):
    n = 400
    x_strong = rng.normal(size=n)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "x_strong": x_strong.round(4),
        "x_strong_copy": (x_strong + rng.normal(0, 0.1, n)).round(4),   # почти дубликат
        "x_weak": rng.normal(size=n).round(4),
        "id_like": rng.integers(0, 1000, n),                            # шум с тысячей значений
        "noise": rng.normal(size=n).round(4),
    })
    df["y"] = (3 * x_strong + df["x_weak"] + rng.normal(0, 1, n)).round(4)
    df["split"] = split_column(n, 0.25, rng)
    perms = pd.DataFrame({f"rep_{r}": rng.permutation(100) for r in range(1, 6)})
    return {"input.csv": df, "permutations.csv": perms}


def data_partial_dependence(rng):
    n = 600
    X = rng.uniform(-2, 2, size=(n, 3)).round(3)
    y = X[:, 0] ** 2 + 2 * X[:, 1] + X[:, 1] * X[:, 2] + rng.normal(0, 0.2, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
                         "y": y.round(4)})


def data_bayesian_optimization(rng):
    x = np.array([-1.6, 0.4, 2.6])
    return pd.DataFrame({"x": x, "f": (np.sin(3 * x) + 0.3 * (x - 1) ** 2).round(6)})


def data_optuna_hyperopt(rng):
    n = 30
    log_lr = rng.uniform(-3, 0, n).round(3)
    subsample = rng.uniform(0.5, 1.0, n).round(3)
    loss = 0.20 + 0.04 * (log_lr + 1.2) ** 2 + 0.3 * (subsample - 0.8) ** 2 + rng.normal(0, 0.003, n)
    trials = pd.DataFrame({"trial": np.arange(1, n + 1), "log_lr": log_lr, "subsample": subsample,
                           "loss": loss.round(4)})
    candidates = pd.DataFrame({"candidate": np.arange(1, 25),
                               "log_lr": rng.uniform(-3, 0, 24).round(3),
                               "subsample": rng.uniform(0.5, 1.0, 24).round(3)})
    return {"input.csv": trials, "candidates.csv": candidates}


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    # ------------------------------------------------------------ Путь 1, этап 2
    {
        "name": "bias_variance",
        "title": "Разложение ошибки на смещение и дисперсию",
        "data": data_bias_variance,
        "task": """
            Истинная зависимость известна: f(x) = sin(2πx), шум ε ~ N(0, 0.3²), т.е. σ² = 0.09.
            Файл `input.csv` содержит **40 независимых обучающих выборок** по 30 точек: колонки `sample` (1–40), `x`, `y`.
            Файл `grid.csv` — 101 контрольная точка: `x` и истинное значение `f`.

            Оцените экспериментально смещение и дисперсию дерева решений для глубин **1, 2, 3, 4, 6, 8, 12**:

            1. Для каждой глубины и каждой выборки обучите `DecisionTreeRegressor(max_depth=глубина, random_state=0)`
               на `x → y` этой выборки и предскажите значения во всех 101 точках сетки.
               Получится матрица прогнозов 40 × 101.
            2. Средний прогноз в точке: f̄(x) = среднее по 40 моделям.
            3. **bias2** = среднее по сетке от (f̄(x) − f(x))².
            4. **variance** = среднее по сетке от дисперсии 40 прогнозов в точке (`ddof=0`).
            5. **expected_mse** = bias2 + variance + σ² — ожидаемая ошибка на новых данных.

            ## Подсказки

            - Смещение — про *средний* прогноз, дисперсия — про *разброс* прогнозов разных моделей в одной точке.
              Обе величины считаются по оси «модели» (40 строк), а потом усредняются по сетке.
            - Соберите прогнозы в массив `preds` формы (40, 101): `preds.mean(axis=0)` — средний прогноз,
              `preds.var(axis=0)` — дисперсия в каждой точке (у NumPy `ddof=0` по умолчанию).
            - Проверка себя: среднее по моделям и точкам от (f̂ − f)² должно точно совпасть с bias2 + variance.
              С ростом глубины bias2 падает, variance растёт, а expected_mse имеет форму буквы U.

            ## Ожидаемый формат выходного файла

            CSV с колонками `max_depth, bias2, variance, expected_mse` — 7 строк, глубины по возрастанию.
            Значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.tree import DecisionTreeRegressor

            DEPTHS = [1, 2, 3, 4, 6, 8, 12]
            SIGMA2 = 0.3 ** 2          # неустранимый шум


            def main():
                df = pd.read_csv('input.csv')
                grid = pd.read_csv('grid.csv')
                x_grid = grid[['x']].to_numpy()
                f_grid = grid['f'].to_numpy()
                samples = [g for _, g in df.groupby('sample')]

                rows = []
                for depth in DEPTHS:
                    # 40 моделей, обученных на разных выборках, — 40 прогнозов в каждой точке сетки
                    preds = np.array([
                        DecisionTreeRegressor(max_depth=depth, random_state=0)
                        .fit(s[['x']].to_numpy(), s['y'].to_numpy()).predict(x_grid)
                        for s in samples
                    ])
                    mean_pred = preds.mean(axis=0)
                    bias2 = np.mean((mean_pred - f_grid) ** 2)
                    variance = preds.var(axis=0).mean()

                    # Разложение точное: E[(f̂ − f)²] = bias² + variance
                    assert np.isclose(np.mean((preds - f_grid) ** 2), bias2 + variance)
                    rows.append({'max_depth': depth, 'bias2': bias2, 'variance': variance,
                                 'expected_mse': bias2 + variance + SIGMA2})

                result = pd.DataFrame(rows).round(4)
                best = result.loc[result['expected_mse'].idxmin(), 'max_depth']
                print(f"Минимум ожидаемой ошибки — при глубине {best}")
                return result
        """,
        "review": """
            **Идея.** Смещение и дисперсию нельзя измерить на одной модели: это свойства *процедуры обучения*,
            а ожидание берётся по разным обучающим выборкам. Поэтому мы обучаем 40 деревьев на 40 выборках и смотрим,
            насколько их средний прогноз далёк от истины (bias²) и насколько прогнозы разбросаны вокруг своего среднего
            (variance). Глубина 1 — две ступеньки на синусоиде, смещение огромное; глубина 12 на 30 точках — дерево
            запоминает каждую точку, смещение почти ноль, зато каждая модель «гуляет» вместе со своим шумом.

            **Типичные ошибки:**
            - Посчитали дисперсию по сетке (по оси точек), а не по моделям — получили разброс самой функции.
            - Сначала усреднили (f̂ − f) по моделям и уже потом не возвели в квадрат как надо — перепутали bias² и MSE.
            - Обучили одну модель на всех 1200 точках сразу — дисперсии между моделями тогда просто нет.
            - Забыли σ² в expected_mse: шум не зависит от модели, но входит в ошибку на новых данных.

            **Что вынести:** ошибка = bias² + variance + σ²; сложность модели двигает баланс между первыми двумя,
            а σ² — нижняя граница. → шпаргалка «Смещение-дисперсия».
        """,
    },
    {
        "name": "overfitting_underfitting",
        "title": "Кривые обучения: недообучение и переобучение",
        "data": data_overfitting_underfitting,
        "task": """
            Дан файл `input.csv`: признак `x`, целевая `y` и `split` — 400 строк `train` и 200 строк `val`.
            Зависимость нелинейная: y ≈ sin(2πx) + 0.5x + шум.

            Постройте **кривые обучения по размеру выборки** для трёх моделей:

            | model | модель |
            |---|---|
            | `linear` | `LinearRegression()` |
            | `tree_d3` | `DecisionTreeRegressor(max_depth=3, random_state=0)` |
            | `tree_full` | `DecisionTreeRegressor(random_state=0)` — без ограничения глубины |

            1. Для каждого размера n ∈ {25, 50, 100, 200, 400} возьмите **первые n строк train** (в порядке файла).
            2. Обучите модель на них, посчитайте MSE на этих же n строках (`train_mse`) и на всех 200 строках val (`val_mse`).
            3. `gap` = val_mse − train_mse.

            ## Подсказки

            - Недообучение — высокая ошибка уже на train и малый разрыв; переобучение — почти нулевая ошибка на train
              и большой разрыв. Какая из трёх моделей на что похожа?
            - Первые n строк train: `train.iloc[:n]` (не перемешивайте и не берите случайную подвыборку).
            - Двойной цикл «модель × размер»; `sklearn.base.clone(model)` даёт свежую копию модели на каждый шаг.
              У `tree_full` train_mse равен нулю при любом n — дерево запоминает каждую точку (совпадающих x в данных нет).

            ## Ожидаемый формат выходного файла

            CSV с колонками `model, n_train, train_mse, val_mse, gap` — 15 строк: модели в порядке
            `linear, tree_d3, tree_full`, внутри модели n по возрастанию. Числа округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.base import clone
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            from sklearn.tree import DecisionTreeRegressor

            MODELS = {
                'linear': LinearRegression(),
                'tree_d3': DecisionTreeRegressor(max_depth=3, random_state=0),
                'tree_full': DecisionTreeRegressor(random_state=0),
            }
            SIZES = [25, 50, 100, 200, 400]


            def main():
                df = pd.read_csv('input.csv')
                train, val = df[df['split'] == 'train'], df[df['split'] == 'val']
                X_val, y_val = val[['x']].to_numpy(), val['y'].to_numpy()

                rows = []
                for name, model in MODELS.items():
                    for n in SIZES:
                        part = train.iloc[:n]
                        X, y = part[['x']].to_numpy(), part['y'].to_numpy()
                        fitted = clone(model).fit(X, y)
                        tr = mean_squared_error(y, fitted.predict(X))
                        va = mean_squared_error(y_val, fitted.predict(X_val))
                        rows.append({'model': name, 'n_train': n, 'train_mse': tr, 'val_mse': va, 'gap': va - tr})

                result = pd.DataFrame(rows).round(4)
                print(result.pivot(index='n_train', columns='model', values='val_mse'))
                return result
        """,
        "review": """
            **Идея.** Две ошибки — на train и на val — и то, как они меняются с ростом данных, ставят диагноз:
            - `linear` — прямая на синусоиде: train и val быстро сходятся на высоком уровне (≈ 0.3, втрое выше шума
              σ² = 0.09), больше данных не помогает. Это недообучение, высокое смещение.
            - `tree_full` — train_mse = 0 всегда, а val_mse держится около 0.16 ≈ 2σ²: каждый лист — одна точка
              вместе с её шумом, поэтому к шуму val добавляется шум train. Разрыв почти не сокращается с ростом n —
              модель без ограничений переобучается при любом объёме данных; помогают регуляризация (глубина,
              `min_samples_leaf`) или усреднение многих деревьев.
            - `tree_d3` — 8 ступенек: на малых n разрыв велик, к n = 400 train и val почти совпадают (≈ 0.10)
              и лучше двух других моделей.

            **Типичные ошибки:**
            - Взяли случайные n строк вместо первых — кривые другие (и не воспроизводятся).
            - Считали train_mse по всем 400 строкам train, а не по тем n, на которых обучали.
            - Переиспользовали один объект модели без `clone` — само по себе не ошибка (`fit` переобучает заново),
              но легко случайно сохранить ссылку на последнюю обученную модель.
            - Подбирали «лучшую модель» по test — для выбора есть val, test трогают один раз в конце.

            **Что вынести:** недообучение лечится усложнением модели и признаками, переобучение — данными,
            регуляризацией и упрощением. Кривые обучения показывают, какой из случаев ваш. → шпаргалка
            «Переобучение и недообучение».
        """,
    },
    # ------------------------------------------------------------ Путь 1, этап 4
    {
        "name": "polynomial_regression",
        "title": "Полиномиальная регрессия: выбор степени по кросс-валидации",
        "data": data_polynomial_regression,
        "task": """
            Дан файл `input.csv`: два признака `x1, x2` и целевая `y` (80 строк).
            Зависимость нелинейная, со взаимодействием признаков.

            Для каждой степени d = 1, 2, …, 6 соберите конвейер

            `make_pipeline(PolynomialFeatures(degree=d, include_bias=False), StandardScaler(), LinearRegression())`

            и посчитайте:

            1. `n_features` — число столбцов после `PolynomialFeatures` (без константы).
            2. `train_mse` — MSE модели, обученной на всех 80 строках, на этих же строках.
            3. `cv_mse` — средняя MSE по 5 фолдам кросс-валидации с `KFold(n_splits=5, shuffle=True, random_state=0)`.

            ## Подсказки

            - Ошибка на train с ростом степени только падает — выбирать степень по ней бессмысленно. Смотрите на cv_mse.
            - Число признаков проверяется формулой: C(n + d, d) − 1, где n = 2 исходных признака (минус 1 — без константы);
              `pipe[0].n_output_features_` после `fit` даёт то же число.
            - `cross_val_score(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring='neg_mean_squared_error')`
              возвращает **отрицательные** MSE по фолдам — возьмите среднее со знаком минус.

            ## Ожидаемый формат выходного файла

            CSV с колонками `degree, n_features, train_mse, cv_mse` — 6 строк, степени по возрастанию.
            Ошибки округлить до 4 знаков.
        """,
        "solution": """
            from math import comb

            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler


            def main():
                df = pd.read_csv('input.csv')
                X, y = df[['x1', 'x2']].to_numpy(), df['y'].to_numpy()
                cv = KFold(n_splits=5, shuffle=True, random_state=0)

                rows = []
                for d in range(1, 7):
                    pipe = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                                         StandardScaler(), LinearRegression())
                    pipe.fit(X, y)
                    n_features = pipe[0].n_output_features_
                    assert n_features == comb(2 + d, d) - 1           # C(n + d, d) без константы
                    train_mse = mean_squared_error(y, pipe.predict(X))
                    cv_mse = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
                    rows.append({'degree': d, 'n_features': n_features, 'train_mse': train_mse, 'cv_mse': cv_mse})

                result = pd.DataFrame(rows)
                print("Лучшая степень по CV:", int(result.loc[result['cv_mse'].idxmin(), 'degree']))
                print(make_pipeline(PolynomialFeatures(3, include_bias=False)).fit(X)[0]
                      .get_feature_names_out(['x1', 'x2']))
                return result.round(4)
        """,
        "review": """
            **Идея.** Полиномиальная регрессия — это всё та же линейная регрессия, но на новых признаках
            x1, x2, x1², x1·x2, x2², …. Данные порождены многочленом 3-й степени (есть x1·x2 и x2³), поэтому
            cv_mse резко падает до d = 3, а дальше растёт: лишние степени подгоняются под шум.
            train_mse при этом монотонно убывает — классическая картина переобучения.

            **Типичные ошибки:**
            - Выбрали степень по train_mse — всегда побеждает максимальная степень.
            - Забыли `shuffle=True, random_state=0` в KFold — фолды другие, cv_mse не совпадает.
            - Взяли `cross_val_score(...).mean()` без минуса — MSE получилась отрицательной.
            - Посчитали n_features с константой (`include_bias=True`) — на 1 больше.
            - Масштабировали признаки до `PolynomialFeatures` и вне конвейера — на CV появляется утечка, а без
              масштабирования x2⁶ доходит до 64 и плохо обусловливает задачу.

            **Что вынести:** степень полинома — гиперпараметр сложности; выбирается по кросс-валидации,
            а число признаков растёт как C(n + d, d). → шпаргалка «Полиномиальная регрессия».
        """,
    },
    {
        "name": "ridge_lasso_elasticnet",
        "title": "Ridge, Lasso и ElasticNet: путь регуляризации",
        "data": data_ridge_lasso_elasticnet,
        "task": """
            Дан файл `input.csv`: 20 признаков `x1…x20`, целевая `y` и `split`. В train всего 30 строк, а признаков 20 —
            обычной регрессии тесно. Полезны лишь несколько признаков, а `x4` — почти копия `x1`.

            1. Масштабируйте признаки `StandardScaler`, **обученным на train**.
            2. Обучите на train и оцените на test (R²):
               - `LinearRegression()` — строка с `model = linear`, `alpha = 0`;
               - `Ridge(alpha=a)`, `Lasso(alpha=a, max_iter=100000)` и `ElasticNet(alpha=a, l1_ratio=0.5, max_iter=100000)`
                 для каждого a ∈ {0.01, 0.1, 1, 10} — строки `ridge`, `lasso`, `elasticnet`.
            3. Для каждой модели запишите `n_nonzero` — число весов с |w| > 1e-6, и `l2_norm` — ‖w‖₂ = √(Σwⱼ²)
               (без intercept).

            ## Подсказки

            - Чем больше alpha, тем сильнее веса прижимаются к нулю. Кто из трёх умеет делать веса *ровно* нулевыми?
            - Удобно собрать модели в список пар (имя, модель) и пройти один цикл; R² — `model.score(X_test, y_test)`.
            - `n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))`, `l2_norm = np.linalg.norm(model.coef_)`.
              Обратите внимание: в sklearn у Lasso штраф делится иначе, чем у Ridge (MSE с множителем 1/(2n)),
              поэтому одинаковые alpha у них — не одинаковая сила регуляризации.

            ## Ожидаемый формат выходного файла

            CSV с колонками `model, alpha, n_nonzero, l2_norm, test_r2` — 13 строк в порядке: `linear`, затем `ridge`
            для alpha по возрастанию, затем `lasso`, затем `elasticnet`. Числа округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
            from sklearn.preprocessing import StandardScaler

            ALPHAS = [0.01, 0.1, 1.0, 10.0]


            def main():
                df = pd.read_csv('input.csv')
                features = [f'x{j}' for j in range(1, 21)]
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                scaler = StandardScaler().fit(train[features])          # только train
                X_train, X_test = scaler.transform(train[features]), scaler.transform(test[features])
                y_train, y_test = train['y'].to_numpy(), test['y'].to_numpy()

                models = [('linear', 0.0, LinearRegression())]
                models += [('ridge', a, Ridge(alpha=a)) for a in ALPHAS]
                models += [('lasso', a, Lasso(alpha=a, max_iter=100000)) for a in ALPHAS]
                models += [('elasticnet', a, ElasticNet(alpha=a, l1_ratio=0.5, max_iter=100000)) for a in ALPHAS]

                rows = []
                for name, alpha, model in models:
                    model.fit(X_train, y_train)
                    rows.append({
                        'model': name, 'alpha': alpha,
                        'n_nonzero': int(np.sum(np.abs(model.coef_) > 1e-6)),
                        'l2_norm': np.linalg.norm(model.coef_),
                        'test_r2': model.score(X_test, y_test),
                    })
                    if name == 'lasso' and alpha == 0.1:
                        kept = [f for f, w in zip(features, model.coef_) if abs(w) > 1e-6]
                        print('Lasso(alpha=0.1) оставил признаки:', kept)

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Все три модели — линейная регрессия со штрафом за большие веса. L2-штраф (Ridge) равномерно
            сжимает все веса, но не зануляет их: n_nonzero всегда 20, а ‖w‖₂ плавно падает с ростом alpha. L1-штраф
            (Lasso) давит на вес с постоянной силой, поэтому бесполезные признаки получают ровно ноль — модель сама
            отбирает признаки. Из почти одинаковых x1 и x4 Lasso при сильном штрафе оставляет только один — какой
            именно, решает случай (здесь при alpha = 1 выживает копия x4), а ElasticNet благодаря L2-части делит вес
            между ними. При alpha = 10 Lasso и ElasticNet обнуляют всё и предсказывают среднее train — R² ниже нуля.
            Лучший R² на test здесь у умеренного Lasso: полезных признаков всего три.

            **Типичные ошибки:**
            - Обучили scaler на всех данных — утечка: среднее и std test влияют на модель.
            - Не масштабировали признаки вовсе — штраф начинает зависеть от единиц измерения.
            - Считали нулевыми веса через `== 0` у Ridge и ждали нулей — их там не бывает, только маленькие числа.
            - Перепутали `l1_ratio`: 1 — чистый Lasso, 0 — чистый Ridge.
            - Включили intercept в ‖w‖ — он не штрафуется.

            **Что вынести:** Ridge — когда полезны многие признаки, Lasso — когда полезны немногие, ElasticNet — когда
            признаков много и они коррелируют. Силу alpha подбирают по CV (`RidgeCV`, `LassoCV`, `ElasticNetCV`).
            → шпаргалка «Ридж, Лассо, ElasticNet».
        """,
    },
    # ------------------------------------------------------------ Путь 1, этап 5
    {
        "name": "confusion_matrix",
        "title": "Многоклассовая матрица ошибок вручную",
        "data": data_confusion_matrix,
        "task": """
            Дан файл `input.csv` с 60 предсказаниями классификатора животных: настоящий класс `y_true`
            и прогноз `y_pred`, классы — `кот`, `пёс`, `лиса`.

            1. Постройте матрицу ошибок 3 × 3 **вручную** (без `sklearn.metrics` — им только проверьте себя):
               строки — настоящие классы, столбцы — предсказанные, порядок классов **кот, пёс, лиса**.
            2. Нормируйте каждую строку на её сумму (как `normalize='true'`): `share_<класс>` — доля объектов
               данного настоящего класса, отнесённых к классу-столбцу.
            3. Для каждого класса (схема «один против всех») найдите TP, FP, FN, TN и посчитайте
               precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP), F1 = 2·P·R/(P+R).

            ## Подсказки

            - Recall и specificity считаются по строкам (от истины), precision — по столбцу (от прогноза).
            - Для класса k: TP = M[k, k], FP = сумма столбца k − TP, FN = сумма строки k − TP, TN = всего − TP − FP − FN.
            - Будьте внимательны с порядком: `sorted(['кот', 'пёс', 'лиса'])` даёт `['кот', 'лиса', 'пёс']`, а нужен
              именно порядок из условия. В sklearn его задают параметром `labels=[...]`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `class, support, share_кот, share_пёс, share_лиса, precision, recall, specificity, f1` —
            3 строки в порядке `кот, пёс, лиса`; `support` — число объектов класса. Дроби округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

            LABELS = ['кот', 'пёс', 'лиса']


            def main():
                df = pd.read_csv('input.csv')
                index = {c: i for i, c in enumerate(LABELS)}
                t = df['y_true'].map(index).to_numpy()
                p = df['y_pred'].map(index).to_numpy()

                M = np.zeros((3, 3), dtype=int)
                np.add.at(M, (t, p), 1)                     # M[истина, прогноз] += 1
                assert np.array_equal(M, confusion_matrix(df['y_true'], df['y_pred'], labels=LABELS))
                print(pd.DataFrame(M, index=LABELS, columns=LABELS))

                total = M.sum()
                rows = []
                for k, name in enumerate(LABELS):
                    tp = M[k, k]
                    fp = M[:, k].sum() - tp                 # столбец: предсказали k, а это не k
                    fn = M[k, :].sum() - tp                 # строка: это k, а предсказали другое
                    tn = total - tp - fp - fn
                    precision, recall = tp / (tp + fp), tp / (tp + fn)
                    row = {'class': name, 'support': int(M[k].sum())}
                    row.update({f'share_{c}': M[k, j] / M[k].sum() for j, c in enumerate(LABELS)})
                    row.update({'precision': precision, 'recall': recall, 'specificity': tn / (tn + fp),
                                'f1': 2 * precision * recall / (precision + recall)})
                    rows.append(row)

                result = pd.DataFrame(rows)
                prec, rec, f1, _ = precision_recall_fscore_support(df['y_true'], df['y_pred'], labels=LABELS)
                assert np.allclose(result['precision'], prec) and np.allclose(result['f1'], f1)
                return result.round(4)
        """,
        "review": """
            **Идея.** Матрица ошибок — таблица «правда × прогноз». Вся многоклассовая метрика сводится к ней:
            для каждого класса считаем «один против всех», и TP/FP/FN/TN берутся из его строки и столбца.
            Нормированная по строкам матрица на диагонали даёт recall каждого класса, а вне диагонали показывает,
            *куда* уходят ошибки: кот и пёс путаются друг с другом, лису иногда принимают за пса, но никогда — за кота.

            **Типичные ошибки:**
            - Перепутали строки и столбцы (истина ↔ прогноз) — precision и recall меняются местами.
            - Отсортировали классы по алфавиту — строки `пёс` и `лиса` поменялись местами.
            - Нормировали по столбцам (`normalize='pred'`) — на диагонали получился precision, а не recall.
            - Считали TN как M[0, 0] для всех классов — в многоклассовом случае TN — это всё, что вне строки и столбца k.

            **Что вынести:** precision — по столбцу, recall и specificity — по строке; нормированная матрица
            подсказывает, какие пары классов модель путает. → шпаргалка «Матрица ошибок».
        """,
    },
    # ------------------------------------------------------------ Путь 1, этап 6
    {
        "name": "boosting",
        "title": "Градиентный бустинг для классификации вручную",
        "data": data_boosting,
        "task": """
            Дан файл `input.csv`: признаки `x1, x2`, бинарная метка `y` и `split` (`train`/`test`).
            Реализуйте **градиентный бустинг с log loss** вручную:

            1. Начальный счёт — логарифм шансов доли единиц в train: F₀ = ln(p̄ / (1 − p̄)); он одинаков для всех объектов.
            2. На каждой итерации m = 1…50:
               - вероятности p = σ(F) = 1 / (1 + e^(−F)) на train;
               - псевдоостатки r = y − p (антиградиент log loss по счёту F);
               - обучите `DecisionTreeRegressor(max_depth=2, random_state=0)` на (X_train, r);
               - обновите счёт на train и на test: F ← F + η·h(x), η = **0.3**.
            3. После итераций **0, 1, 5, 10, 20, 30, 40, 50** (0 — только F₀) запишите log loss на train и test
               и accuracy на test (класс 1, если p > 0.5).

            Log loss = −mean(y·ln p + (1 − y)·ln(1 − p)).

            ## Подсказки

            - Модель выдаёт не вероятность, а счёт F — логарифм шансов, как в логистической регрессии. Деревья
              добавляются к счёту, а вероятность получается сигмоидой в самом конце.
            - Счёт на test обновляйте теми же деревьями, что обучены на train, — test в обучении не участвует.
            - Цикл: `p = 1 / (1 + np.exp(-F_train))`, `tree = DecisionTreeRegressor(...).fit(X_train, y_train - p)`,
              `F_train += 0.3 * tree.predict(X_train)`, `F_test += 0.3 * tree.predict(X_test)`.
              Проверка: на итерации 0 log loss на train равен энтропии −(p̄·ln p̄ + (1−p̄)·ln(1−p̄)).

            ## Ожидаемый формат выходного файла

            CSV с колонками `n_trees, train_logloss, test_logloss, test_accuracy` — 8 строк для
            n_trees = 0, 1, 5, 10, 20, 30, 40, 50. Значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import log_loss
            from sklearn.tree import DecisionTreeRegressor

            LR, N_TREES = 0.3, 50
            REPORT = {0, 1, 5, 10, 20, 30, 40, 50}


            def sigmoid(z):
                return 1 / (1 + np.exp(-z))


            def logloss(y, p):
                p = np.clip(p, 1e-15, 1 - 1e-15)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                X_train, y_train = train[['x1', 'x2']].to_numpy(), train['y'].to_numpy()
                X_test, y_test = test[['x1', 'x2']].to_numpy(), test['y'].to_numpy()

                p_bar = y_train.mean()
                F_train = np.full(len(y_train), np.log(p_bar / (1 - p_bar)))   # F₀ — логарифм шансов
                F_test = np.full(len(y_test), F_train[0])

                rows = []
                for m in range(N_TREES + 1):
                    if m > 0:
                        residual = y_train - sigmoid(F_train)                  # антиградиент log loss
                        tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_train, residual)
                        F_train += LR * tree.predict(X_train)
                        F_test += LR * tree.predict(X_test)
                    if m in REPORT:
                        p_test = sigmoid(F_test)
                        rows.append({'n_trees': m,
                                     'train_logloss': logloss(y_train, sigmoid(F_train)),
                                     'test_logloss': logloss(y_test, p_test),
                                     'test_accuracy': np.mean((p_test > 0.5) == y_test)})

                assert np.isclose(rows[-1]['test_logloss'], log_loss(y_test, sigmoid(F_test)))
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Бустинг — градиентный спуск в пространстве функций. Для log loss антиградиент по счёту равен
            y − p: каждое новое дерево учится на том, насколько текущая вероятность расходится с правильным ответом,
            и сдвигает счёт на η своей поправки. Train log loss падает монотонно и быстро, а test — всё медленнее,
            и разрыв между ними растёт: поздние деревья всё больше подгоняют шум. Если продолжать, test перестанет
            улучшаться — момент остановки ищут по валидации (ранняя остановка).

            **Типичные ошибки:**
            - Добавляли деревья к вероятности, а не к счёту — p выходит за [0, 1].
            - Начали с F₀ = p̄ вместо логарифма шансов — на итерации 0 вероятность σ(p̄) ≠ p̄.
            - Учили деревья на остатках test или обновляли F_test деревьями, обученными заново на test.
            - Забыли η — модель за пару шагов «перепрыгивает» и быстро переобучается.

            **Подробнее.** В `GradientBoostingClassifier` значения листьев после построения дерева заменяются
            шагом Ньютона Σr / Σp(1 − p) — это тот же приём, что в XGBoost (см. упражнение `xgboost`). Мы оставили
            среднее остатков, как в лекции, поэтому числа немного отличаются от sklearn.

            **Что вынести:** F = F₀ + η·Σhₘ, деревья учатся на псевдоостатках, вероятность — σ(F);
            learning rate и число деревьев связаны, а лучшее число деревьев ищут по валидации.
            → шпаргалка «Градиентный бустинг».
        """,
    },
    {
        "name": "xgboost",
        "title": "XGBoost вручную: градиенты, гессианы, прирост и веса листьев",
        "data": data_xgboost,
        "task": """
            Дан файл `input.csv`: признаки `x1` (вещественный), `x2` (целый), бинарная метка `y` и `F` — счёт (логарифм шансов),
            который выдают уже построенные деревья ансамбля. Постройте **следующее дерево XGBoost глубины 2** для log loss
            вручную, на NumPy.

            1. p = σ(F); градиент gᵢ = pᵢ − yᵢ, гессиан hᵢ = pᵢ(1 − pᵢ).
            2. Для узла с суммами G = Σg, H = Σh: вес листа w* = −G / (H + λ), λ = **1**.
            3. Кандидаты разбиения — по каждому признаку середины между соседними **уникальными** отсортированными значениями
               в узле; влево идут объекты с x **<** порога.
               Gain = ½·[G_L²/(H_L + λ) + G_R²/(H_R + λ) − G²/(H + λ)] − γ, γ = **0.1**.
            4. В каждом узле выберите разбиение с максимальным Gain (перебор: сначала `x1`, затем `x2`, пороги по возрастанию;
               при равенстве остаётся первый). Корень делится, затем делятся оба его потомка; их потомки — листья.

            Нумерация узлов — по уровням: 0 — корень, 1 и 2 — левый и правый потомки корня, 3 и 4 — потомки узла 1, 5 и 6 — узла 2.

            ## Подсказки

            - Gain показывает, насколько разбиение уменьшает целевую функцию второго порядка; λ «сжимает» вклад узлов
              с малым H, а γ — цена нового листа.
            - Напишите функцию `best_split(X, g, h)`, которая перебирает признаки и пороги и возвращает (признак, порог, gain);
              вызовите её для корня, а затем для левого и правого подмножеств.
            - Порог: `v = np.unique(x)`, `thresholds = (v[:-1] + v[1:]) / 2`, маска `left = x < thr`. Суммы G_L, H_L — по маске,
              G_R = G − G_L, H_R = H − H_L. Для листа: `w = -G / (H + 1)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `node, depth, n, G, H, weight, feature, threshold, gain` — 7 строк, узлы 0…6 по порядку.
            `n` — число объектов в узле, `weight` — w* узла (у внутренних узлов тоже). Для внутренних узлов `feature` —
            `x1` или `x2`, `threshold` и `gain` — выбранного разбиения; для листьев `feature = leaf`, а `threshold` и
            `gain` пустые. Числа округлить до 4 знаков.
        """,
        "solution": """
            LAMBDA, GAMMA = 1.0, 0.1
            FEATURES = ['x1', 'x2']


            def score(G, H):
                return G ** 2 / (H + LAMBDA)


            def best_split(X, g, h):
                G, H = g.sum(), h.sum()
                candidates = []
                for j, name in enumerate(FEATURES):
                    values = np.unique(X[:, j])
                    for thr in (values[:-1] + values[1:]) / 2:
                        left = X[:, j] < thr
                        GL, HL = g[left].sum(), h[left].sum()
                        gain = 0.5 * (score(GL, HL) + score(G - GL, H - HL) - score(G, H)) - GAMMA
                        candidates.append((gain, name, j, thr))
                best = max(candidates, key=lambda c: c[0])        # max берёт первый из равных
                gains = sorted((c[0] for c in candidates), reverse=True)
                assert gains[0] - gains[1] > 1e-6                 # победитель однозначен
                return best


            def main():
                df = pd.read_csv('input.csv')
                X = df[FEATURES].to_numpy(dtype=float)
                p = 1 / (1 + np.exp(-df['F'].to_numpy()))
                g = p - df['y'].to_numpy()                        # градиент log loss по счёту
                h = p * (1 - p)                                   # гессиан

                nodes = {0: np.arange(len(df))}                   # номер узла → индексы объектов
                rows = []
                for node in range(7):
                    idx = nodes[node]
                    G, H = g[idx].sum(), h[idx].sum()
                    row = {'node': node, 'depth': int(np.log2(node + 1)), 'n': len(idx),
                           'G': G, 'H': H, 'weight': -G / (H + LAMBDA)}
                    if node < 3:                                  # внутренние узлы: корень и его потомки
                        gain, name, j, thr = best_split(X[idx], g[idx], h[idx])
                        assert gain > 0, 'разбиение должно окупать γ'
                        left = X[idx, j] < thr
                        nodes[2 * node + 1], nodes[2 * node + 2] = idx[left], idx[~left]
                        row.update({'feature': name, 'threshold': thr, 'gain': gain})
                    else:
                        row.update({'feature': 'leaf', 'threshold': np.nan, 'gain': np.nan})
                    rows.append(row)

                result = pd.DataFrame(rows)
                print(result.round(4))
                return result.round(4)
        """,
        "review": """
            **Идея.** XGBoost раскладывает потери в ряд Тейлора до второго порядка: в каждом объекте известны
            наклон g и кривизна h. Тогда оптимальный вес листа и прирост от разбиения считаются в явном виде —
            через суммы G и H, без всякой подгонки. w* = −G/(H + λ) — это шаг Ньютона, «сжатый» L2-регуляризацией,
            а Gain — сколько целевой функции мы выигрываем, разделив узел, минус цена нового листа γ.

            **Типичные ошибки:**
            - Взяли g = y − p (знак псевдоостатка из обычного бустинга) — веса листьев поменяли знак.
            - Положили h = 1 для всех объектов — получился бустинг по MSE, а не по log loss.
            - Забыли ½ или γ в формуле Gain — победитель может совпасть, но значения gain не совпадут.
            - Перебирали пороги по всем значениям, а не по серединам уникальных, или отправляли x ≤ порога влево:
              при целых `x2` это сдвигает разбиение на одно значение.

            **Что вынести:** gᵢ и hᵢ — всё, что дерево XGBoost знает о функции потерь; λ уменьшает веса малых листьев,
            γ обрезает бесполезные разбиения прямо при построении дерева. → шпаргалка «XGBoost».
        """,
    },
    {
        "name": "lightgbm",
        "title": "LightGBM вручную: гистограммы и поиск разбиения по корзинам",
        "data": data_lightgbm,
        "task": """
            Дан файл `input.csv`: признаки `x1, x2` и целевая `y` (400 строк). Повторите главный приём LightGBM —
            **поиск лучшего разбиения по гистограммам**, а не по всем уникальным значениям.

            1. Начальный прогноз F₀ = mean(y). Для потерь ½(F − y)²: градиент gᵢ = F₀ − yᵢ, гессиан hᵢ = 1.
            2. Для каждого признака разбейте значения на **8 корзин по квантилям**: границы
               `edges = np.quantile(x, np.linspace(0, 1, 9))` (линейная интерполяция, по умолчанию в NumPy);
               номер корзины = число внутренних границ `edges[1:-1]`, которые ≤ x, т.е. от 0 до 7.
            3. Гистограмма признака: для каждой корзины — число объектов `count` и сумма градиентов `grad_sum`.
            4. Разбиение «после корзины k» отправляет влево корзины 0…k. Его прирост
               gain = G_L²/(H_L + λ) + G_R²/(H_R + λ) − G²/(H + λ), λ = **1**, где G, H — суммы g и h.
               Для последней корзины (k = 7) разбиения нет — запишите gain = 0.

            ## Подсказки

            - Найти лучшее разбиение можно, не сортируя 400 значений: достаточно накопленных сумм по 8 корзинам.
              Для H при h = 1 сумма гессианов — это просто число объектов.
            - Номер корзины: `np.searchsorted(edges[1:-1], x, side='right')`. Гистограмма:
              `np.bincount(bins, weights=g, minlength=8)` и `np.bincount(bins, minlength=8)`.
            - G_L для всех k сразу — `np.cumsum(grad_sum)`, H_L — `np.cumsum(count)`; G_R = G − G_L, H_R = H − H_L.
              Проверьте себя: сумма `grad_sum` по корзинам равна нулю (F₀ — среднее), поэтому G²/(H + λ) ≈ 0.

            ## Ожидаемый формат выходного файла

            CSV с колонками `feature, bin, count, grad_sum, gain` — 16 строк: сначала 8 корзин `x1` (bin 0…7),
            затем 8 корзин `x2`. `grad_sum` и `gain` округлить до 4 знаков.
        """,
        "solution": """
            N_BINS, LAMBDA = 8, 1.0
            FEATURES = ['x1', 'x2']


            def gain_formula(GL, HL, G, H):
                GR, HR = G - GL, H - HL
                return GL ** 2 / (HL + LAMBDA) + GR ** 2 / (HR + LAMBDA) - G ** 2 / (H + LAMBDA)


            def main():
                df = pd.read_csv('input.csv')
                y = df['y'].to_numpy()
                g = y.mean() - y                      # градиент ½(F − y)² в точке F₀ = mean(y)
                h = np.ones_like(g)                   # гессиан
                G, H = g.sum(), h.sum()

                rows = []
                for name in FEATURES:
                    x = df[name].to_numpy()
                    edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
                    bins = np.searchsorted(edges[1:-1], x, side='right')       # 0 … 7

                    count = np.bincount(bins, minlength=N_BINS)
                    grad_sum = np.bincount(bins, weights=g, minlength=N_BINS)
                    hess_sum = np.bincount(bins, weights=h, minlength=N_BINS)

                    GL, HL = np.cumsum(grad_sum), np.cumsum(hess_sum)          # влево — корзины 0…k
                    gain = gain_formula(GL, HL, G, H)
                    gain[-1] = 0.0                                             # после последней корзины делить нечего

                    # Для сравнения: точный перебор всех порогов (как без гистограмм)
                    order = np.argsort(x)
                    exact = gain_formula(np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1], G, H)
                    print(f"{name}: лучший gain по 8 корзинам = {gain.max():.2f}, "
                          f"точный по {len(x) - 1} порогам = {exact.max():.2f}")

                    rows += [{'feature': name, 'bin': k, 'count': int(count[k]),
                              'grad_sum': grad_sum[k], 'gain': gain[k]} for k in range(N_BINS)]

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Чтобы найти лучший порог, бустингу нужны лишь суммы градиентов и гессианов слева и справа.
            Если заранее разложить значения по корзинам, сканировать придётся 8 (в LightGBM — до 255) корзин вместо
            всех уникальных значений, а гистограммы считаются за один проход. Лучшее разбиение по `x1` — после корзины 4,
            на границе около x1 ≈ 6.25, рядом с настоящим скачком y на 4 при x1 = 6.3. Прирост по 8 корзинам лишь
            на ~10% меньше точного по 399 порогам (в выводе решения), а с 255 корзинами разница почти исчезает.
            По `x2` эффект слабый, gain на порядок меньше — дерево выберет `x1`.

            **Типичные ошибки:**
            - Корзины равной ширины вместо квантильных — counts и границы другие.
            - `side='left'` или `pd.cut` с правыми закрытыми интервалами — объекты на границах уходят в соседнюю корзину
              (здесь значения на границы не попадают, но в общем случае это важно).
            - Взяли g = y − F₀ — знак grad_sum противоположный (gain от знака не зависит).
            - Забыли λ в знаменателе: при h = 1 это заметно только в маленьких корзинах, но числа не совпадут.

            **Что вынести:** гистограммы — основа скорости LightGBM (и `HistGradientBoosting` в sklearn); точность
            почти не теряется, а ещё гистограмму одного потомка можно получить вычитанием из гистограммы родителя.
            → шпаргалка «LightGBM».
        """,
    },
    {
        "name": "catboost",
        "title": "CatBoost вручную: ordered target statistics",
        "data": data_catboost,
        "task": """
            Дан файл `input.csv`: категориальный признак `city`, метка `bought` (купил ли клиент) и `split`.
            Строки train (30 штук) уже идут в порядке **случайной перестановки** — это порядок «времени» для CatBoost.
            Строки test (10 штук) идут после них; их `bought` при кодировании не используется.

            Закодируйте `city` двумя способами, prior = доля `bought = 1` в train, a = **1**:

            1. **ts_ordered** (как в CatBoost) — для train-строки на позиции i:
               TS(i) = (сумма bought у *предыдущих* train-строк с тем же городом + a·prior) / (число таких строк + a).
               Сам объект и все объекты после него не используются.
            2. **ts_greedy** («наивное» target-кодирование) — та же формула, но по **всем** train-строкам города,
               включая саму строку.
            3. Для test обе колонки одинаковы: формула по всем строкам train с этим городом
               (если город в train не встречался — получается prior).

            ## Подсказки

            - Подумайте, почему ts_greedy опасен: значение для строки содержит её собственный `bought`.
              Особенно сильно это видно на редких городах.
            - Удобно идти по train в цикле и держать словари «город → сумма» и «город → число»:
              сначала посчитать TS по текущим значениям, потом добавить строку.
            - Без цикла: `prev_sum = groupby('city')['bought'].cumsum() - bought`,
              `prev_cnt = groupby('city').cumcount()`. Первая строка каждого города получает ровно prior.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, split, ts_ordered, ts_greedy` — 40 строк в порядке файла (сначала train, затем test).
            Значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import roc_auc_score

            A = 1.0   # сила сглаживания


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                prior = train['bought'].mean()

                # Ordered TS: только «прошлые» объекты в перестановке
                sums, counts, ordered = {}, {}, []
                for city, y in zip(train['city'], train['bought']):
                    s, c = sums.get(city, 0.0), counts.get(city, 0)
                    ordered.append((s + A * prior) / (c + A))
                    sums[city], counts[city] = s + y, c + 1

                # Векторная проверка той же формулы
                prev_sum = train.groupby('city')['bought'].cumsum() - train['bought']
                prev_cnt = train.groupby('city').cumcount()
                assert np.allclose(ordered, (prev_sum + A * prior) / (prev_cnt + A))

                # Greedy TS: все train-объекты города, включая сам объект
                stats = train.groupby('city')['bought'].agg(['sum', 'count'])
                greedy = ((train['city'].map(stats['sum']) + A * prior)
                          / (train['city'].map(stats['count']) + A)).to_numpy()

                # Test: статистика по всему train; новый город — prior
                test_ts = ((test['city'].map(stats['sum']).fillna(0) + A * prior)
                           / (test['city'].map(stats['count']).fillna(0) + A)).to_numpy()

                y = train['bought']
                print(f"ROC-AUC кодировки на train: greedy = {roc_auc_score(y, greedy):.3f}, "
                      f"ordered = {roc_auc_score(y, ordered):.3f}  ← greedy «подсматривает» в ответ")

                return pd.DataFrame({
                    'id': df['id'], 'split': df['split'],
                    'ts_ordered': np.concatenate([ordered, test_ts]),
                    'ts_greedy': np.concatenate([greedy, test_ts]),
                }).round(4)
        """,
        "review": """
            **Идея.** Target-кодирование превращает категорию в число — «средний таргет категории». Но если в среднее
            входит сам объект, признак содержит кусочек ответа: на train он кажется сильнее, чем на новых данных
            (это видно по ROC-AUC кодировки в выводе решения). CatBoost решает это как с временным рядом: объект видит
            только тех, кто стоит раньше него в случайной перестановке, а prior сглаживает первые, самые шумные значения.

            **Типичные ошибки:**
            - Включили текущую строку в «предыдущие» (`cumsum` без вычитания `bought`) — получилось почти greedy.
            - Сгладили не к prior, а к 0 или 0.5 — у первых объектов каждого города другие значения.
            - Для test пересчитали prior или статистики с учётом test-меток — утечка.
            - Новый город в test получил NaN вместо prior.

            **Подробнее.** Настоящий CatBoost использует несколько перестановок и ещё ordered boosting — тот же принцип
            «не подсматривать в свой ответ» для остатков самих деревьев.

            **Что вынести:** при target-кодировании объект не должен видеть свой таргет: ordered TS, out-of-fold
            кодирование или leave-one-out. → шпаргалка «CatBoost».
        """,
    },
    # ------------------------------------------------------------ Путь 2, этап 1
    {
        "name": "perceptron",
        "title": "Обучение перцептрона Розенблатта вручную",
        "data": data_perceptron,
        "task": """
            Дан файл `input.csv`: 40 точек с признаками `x1, x2` и классом `y` (0/1). Классы линейно разделимы.
            Обучите **перцептрон** правилом из лекции:

            1. Старт: w = (0, 0), b = 0, скорость обучения η = **0.1**.
            2. Эпоха — один проход по строкам **в порядке файла**. Для каждой строки:
               ŷ = 1, если w·x + b > 0, иначе 0 (при z = 0 ответ 0);
               w ← w + η·(y − ŷ)·x, b ← b + η·(y − ŷ).
            3. После эпохи запишите число ошибок за эпоху (сколько раз было y ≠ ŷ, т.е. обновлений), текущие w₁, w₂, b
               и accuracy на всех 40 точках с весами конца эпохи.
            4. Остановитесь после **первой эпохи без ошибок** (её тоже запишите).

            ## Подсказки

            - Перцептрон меняет веса только на ошибках; если за эпоху ошибок нет, прямая уже разделяет классы
              и дальше ничего не изменится (теорема сходимости).
            - Обновления идут **сразу** после каждой строки (онлайн), а не после всей эпохи.
            - Внутренний цикл: `z = x @ w + b`, `y_hat = int(z > 0)`, `err = y - y_hat`, `w += lr * err * x`, `b += lr * err`,
              `n_errors += err != 0`. Accuracy в конце эпохи: `np.mean(((X @ w + b) > 0) == y)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, n_errors, w1, w2, b, accuracy` — по строке на эпоху (с 1 до первой эпохи
            с `n_errors = 0` включительно). Веса и accuracy округлить до 4 знаков.
        """,
        "solution": """
            LR, MAX_EPOCHS = 0.1, 100


            def main():
                df = pd.read_csv('input.csv')
                X = df[['x1', 'x2']].to_numpy(dtype=float)
                y = df['y'].to_numpy()

                w, b = np.zeros(2), 0.0
                rows = []
                for epoch in range(1, MAX_EPOCHS + 1):
                    n_errors = 0
                    for xi, yi in zip(X, y):
                        z = xi @ w + b
                        # Однозначность: ненулевой z не должен быть «почти нулём» из-за округлений
                        assert z == 0 or abs(z) > 1e-9
                        err = yi - int(z > 0)            # ступенчатая активация
                        w += LR * err * xi               # правило перцептрона
                        b += LR * err
                        n_errors += int(err != 0)
                    acc = np.mean(((X @ w + b) > 0).astype(int) == y)
                    rows.append({'epoch': epoch, 'n_errors': n_errors, 'w1': w[0], 'w2': w[1], 'b': b, 'accuracy': acc})
                    if n_errors == 0:
                        break
                else:
                    raise RuntimeError('перцептрон не сошёлся')

                print(f"Сошёлся за {len(rows)} эпох: {w[0]:.2f}·x1 + {w[1]:.2f}·x2 + {b:.2f} = 0")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Перцептрон — линейный классификатор w·x + b > 0, который учится только на ошибках: если ответил 0
            вместо 1, прямая сдвигается к объекту, если наоборот — от него. Классы здесь линейно разделимы, поэтому
            по теореме сходимости через конечное число эпох ошибок не остаётся. Число ошибок по эпохам не обязано убывать
            монотонно — исправив одну точку, прямая может «задеть» другую.

            **Типичные ошибки:**
            - Обновляли веса после всей эпохи (пакетно) — это уже другой алгоритм, траектория другая.
            - Считали ŷ = 1 при z ≥ 0 — на первой же строке (w = 0) ответ меняется, и вся история весов тоже.
            - Забыли обновлять b — без смещения прямая обязана проходить через начало координат.
            - Перемешали строки — перцептрон чувствителен к порядку объектов.

            **Интересно.** При нулевом старте η лишь масштабирует w и b: знак w·x + b, а значит, все ошибки и номер эпохи
            сходимости от η не зависят. Для XOR же никакая прямая не подходит, и цикл никогда не закончится —
            нужен второй слой (MLP).

            **Что вынести:** нейрон = линейная комбинация + активация; правило перцептрона сходится только для линейно
            разделимых классов. → шпаргалка «Искусственный нейрон (перцептрон)».
        """,
    },
    # ------------------------------------------------------------ Путь 3, этап 1
    {
        "name": "feature_importance",
        "title": "MDI против permutation importance вручную",
        "data": data_feature_importance,
        "task": """
            Дан файл `input.csv`: признаки `x_strong, x_strong_copy, x_weak, id_like, noise`, целевая `y` и `split`.
            `x_strong_copy` — почти точная копия `x_strong`, `id_like` и `noise` — чистый шум.
            Файл `permutations.csv` — 5 готовых перестановок (`rep_1…rep_5`) позиций 0…99 тестовой выборки.

            1. Обучите `RandomForestRegressor(n_estimators=100, random_state=0)` на train (5 признаков в указанном порядке).
            2. `mdi` — `feature_importances_` модели.
            3. **Permutation importance вручную** на test: базовый score — R² на test. Для каждого признака и каждой из 5
               перестановок замените столбец признака в test на `значения[perm]` (остальные столбцы не трогайте),
               посчитайте R² и важность = базовый R² − новый R².
            4. `perm_mean` и `perm_std` — среднее и стандартное отклонение (`ddof=0`) по 5 перестановкам.

            ## Подсказки

            - Перемешивание разрушает связь признака с ответом, но сохраняет его распределение; модель заново не обучается.
            - Перестановка задаёт позиции **внутри test** (в порядке файла): `X_perm[col] = X_test[col].to_numpy()[perm]`.
              Берите копию `X_test` для каждой перестановки, чтобы не испортить исходные данные.
            - Что ожидать: у шумовых признаков perm ≈ 0, а MDI — заметно больше нуля; у пары
              `x_strong` / `x_strong_copy` permutation importance обоих занижена — модель берёт информацию из «близнеца».

            ## Ожидаемый формат выходного файла

            CSV с колонками `feature, mdi, perm_mean, perm_std` — 5 строк в порядке
            `x_strong, x_strong_copy, x_weak, id_like, noise`. Значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score

            FEATURES = ['x_strong', 'x_strong_copy', 'x_weak', 'id_like', 'noise']


            def main():
                df = pd.read_csv('input.csv')
                perms = pd.read_csv('permutations.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                X_test, y_test = test[FEATURES].reset_index(drop=True), test['y'].to_numpy()

                rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(train[FEATURES], train['y'])
                base = r2_score(y_test, rf.predict(X_test))
                print(f"R² на test: {base:.3f}")

                rows = []
                for j, col in enumerate(FEATURES):
                    drops = []
                    for rep in perms.columns:
                        X_perm = X_test.copy()
                        X_perm[col] = X_test[col].to_numpy()[perms[rep].to_numpy()]   # перемешали один столбец
                        drops.append(base - r2_score(y_test, rf.predict(X_perm)))
                    rows.append({'feature': col, 'mdi': rf.feature_importances_[j],
                                 'perm_mean': np.mean(drops), 'perm_std': np.std(drops)})

                # Если убрать «близнеца», важность x_strong резко вырастет
                solo = [f for f in FEATURES if f != 'x_strong_copy']
                rf2 = RandomForestRegressor(n_estimators=100, random_state=0).fit(train[solo], train['y'])
                Xs = test[solo].reset_index(drop=True)
                X_perm = Xs.copy()
                X_perm['x_strong'] = Xs['x_strong'].to_numpy()[perms['rep_1'].to_numpy()]
                drop = r2_score(y_test, rf2.predict(Xs)) - r2_score(y_test, rf2.predict(X_perm))
                print(f"Без x_strong_copy падение R² при перемешивании x_strong: {drop:.3f}")

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** MDI считается по обучающей выборке во время построения деревьев: каждое уменьшение неоднородности
            записывается на счёт признака. Поэтому шум, на котором деревья делали случайные разбиения (особенно
            `id_like` с тысячей значений), получает заметную MDI. Permutation importance честнее: она меряет, сколько
            качества на **test** теряется без информации из признака, и у шума даёт ≈ 0.

            Коррелированная пара показывает слабость обоих методов: MDI делит важность между `x_strong` и копией
            произвольно, а permutation importance занижает обоих — перемешали один, модель взяла то же из второго.
            Решение — перемешивать группу целиком или оставить один признак из пары.

            **Типичные ошибки:**
            - Перемешивали столбец «на месте» и не вернули его — следующие признаки считаются на испорченных данных.
            - Перемешали весь DataFrame строками, а не один столбец, — разрушили связь всех признаков.
            - Применили перестановку к индексам исходного `df` (номера 1…400), а не к позициям 0…99 внутри test.
            - Считали permutation importance на train — она тогда тоже «верит» переобучению.

            **Что вынести:** важность — свойство модели, а не мира; MDI быстрая, но смещённая; permutation importance
            считайте на отложенной выборке и следите за коррелированными признаками. → шпаргалка «Важность признаков».
        """,
    },
    {
        "name": "bayesian_optimization",
        "title": "Байесовская оптимизация: гауссовский процесс и Expected Improvement",
        "data": data_bayesian_optimization,
        "task": """
            Нужно найти минимум «дорогой» функции f(x) = sin(3x) + 0.3·(x − 1)² на отрезке [−2, 3].
            В `input.csv` — 3 уже вычисленные стартовые точки (`x`, `f`). Сделайте **8 шагов байесовской оптимизации**:

            1. Сетка кандидатов: `grid = np.linspace(-2, 3, 201)`.
            2. Суррогат: `GaussianProcessRegressor(kernel=Matern(length_scale=0.5, nu=2.5), alpha=1e-6,
               normalize_y=True, optimizer=None)` — параметры ядра **фиксированы**, не подбираются.
               Обучите его на всех вычисленных точках, получите μ(x) и σ(x) на сетке (`return_std=True`).
            3. Expected Improvement для минимизации, f* — лучшее найденное значение:
               z = (f* − μ) / σ,  EI = (f* − μ)·Φ(z) + σ·φ(z); где σ < 1e-12, считайте EI = 0.
            4. Следующая точка — узел сетки с максимальным EI (при равенстве — первый). Вычислите в ней f и добавьте к данным.

            ## Подсказки

            - Первое слагаемое EI велико там, где прогноз хорош (использование), второе — там, где велика неуверенность
              (исследование). Рядом с уже вычисленными точками σ ≈ 0, и EI там почти ноль.
            - Φ и φ — `scipy.stats.norm.cdf` и `norm.pdf`; номер лучшего узла — `np.argmax(ei)`.
            - GP можно посчитать и вручную: при `normalize_y` y′ = (y − mean)/std (std с `ddof=0`); K = k(X, X) + αI,
              μ′ = k*ᵀK⁻¹y′, σ′² = 1 − k*ᵀK⁻¹k*, затем μ = mean + std·μ′, σ = std·σ′.
              Matern 5/2: k(r) = (1 + √5·r/ℓ + 5r²/(3ℓ²))·exp(−√5·r/ℓ).

            ## Ожидаемый формат выходного файла

            CSV с колонками `step, x_next, ei, f_next, best_f` — 8 строк для шагов 1…8. `ei` — максимум EI на этом шаге,
            `best_f` — лучшее значение f после добавления новой точки. Значения округлить до 4 знаков.
        """,
        "solution": """
            from scipy.stats import norm
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            N_STEPS, ELL = 8, 0.5


            def f(x):
                return np.sin(3 * x) + 0.3 * (x - 1) ** 2


            def matern52(a, b):
                r = np.abs(a[:, None] - b[None, :]) / ELL
                return (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)


            def gp_by_hand(X, y, grid, alpha=1e-6):
                mean, std = y.mean(), y.std()
                yn = (y - mean) / std
                K = matern52(X, X) + alpha * np.eye(len(X))
                Ks = matern52(X, grid)
                mu = Ks.T @ np.linalg.solve(K, yn)
                var = 1 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
                return mean + std * mu, std * np.sqrt(np.maximum(var, 0))


            def main():
                start = pd.read_csv('input.csv')
                X, y = start['x'].to_numpy(), start['f'].to_numpy()
                grid = np.linspace(-2, 3, 201)

                rows = []
                for step in range(1, N_STEPS + 1):
                    gp = GaussianProcessRegressor(kernel=Matern(length_scale=ELL, nu=2.5), alpha=1e-6,
                                                  normalize_y=True, optimizer=None).fit(X[:, None], y)
                    mu, sigma = gp.predict(grid[:, None], return_std=True)
                    mu_h, sigma_h = gp_by_hand(X, y, grid)
                    assert np.allclose(mu, mu_h, atol=1e-6) and np.allclose(sigma, sigma_h, atol=1e-5)

                    f_best = y.min()
                    ei = np.zeros_like(mu)
                    ok = sigma >= 1e-12
                    z = (f_best - mu[ok]) / sigma[ok]
                    ei[ok] = (f_best - mu[ok]) * norm.cdf(z) + sigma[ok] * norm.pdf(z)

                    k = int(np.argmax(ei))
                    second = np.partition(ei, -2)[-2]
                    assert ei[k] - second > 1e-9          # победитель однозначен
                    x_next, f_next = grid[k], f(grid[k])
                    X, y = np.append(X, x_next), np.append(y, f_next)
                    rows.append({'step': step, 'x_next': x_next, 'ei': ei[k], 'f_next': f_next, 'best_f': y.min()})

                print(f"Найдено: x = {X[y.argmin()]:.3f}, f = {y.min():.4f} за {len(y)} вычислений; "
                      f"минимум на сетке: {f(grid).min():.4f}")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Вместо дорогой функции мы оптимизируем дешёвый суррогат. Гауссовский процесс в каждой точке
            даёт прогноз μ и неуверенность σ, а EI превращает их в одно число — «сколько в среднем выиграем, если
            проверим здесь». Первые пять шагов EI спускается во впадину около x ≈ 1.5 (использование) и на 5-м шаге
            находит x = 1.55, f ≈ −0.907 — почти минимум на сетке (−0.908). Когда улучшать рядом уже нечего, EI
            переключается на неисследованный левый край (исследование): вдруг глобальный минимум там.
            Всего 11 вычислений вместо 201 при полном переборе сетки.

            **Типичные ошибки:**
            - Оставили оптимизатор ядра включённым — длина ℓ подбирается под данные, и точки получаются другими.
            - EI для максимизации: (μ − f*) вместо (f* − μ) — алгоритм идёт к максимуму функции.
            - Делили на σ = 0 в уже проверенных точках — NaN в EI и непредсказуемый `argmax`.
            - Подставили в EI дисперсию вместо стандартного отклонения.
            - Не обновили f* после добавления точки — EI считает улучшение относительно старого рекорда.

            **Что вынести:** BO = суррогат + функция выбора; EI сам балансирует исследование и использование.
            GP хорош для нескольких непрерывных параметров и десятков–сотен точек. → шпаргалка «Bayesian Optimization».
        """,
    },
    {
        "name": "optuna_hyperopt",
        "title": "Один шаг TPE вручную (идея Optuna и Hyperopt)",
        "data": data_optuna_hyperopt,
        "task": """
            В `input.csv` — история 30 проб случайного поиска гиперпараметров бустинга: `log_lr` = log₁₀(learning_rate)
            из [−3, 0], `subsample` из [0.5, 1] и ошибка на валидации `loss` (меньше — лучше).
            В `candidates.csv` — 24 кандидата, из которых TPE должен выбрать следующую пробу.

            Сделайте **один шаг TPE** (упрощённого, с фиксированной шириной ядер):

            1. «Хорошие» пробы — **6** с наименьшим `loss` (γ = 20%), «плохие» — остальные 24.
            2. Для каждого параметра отдельно постройте две оценки плотности Парзена — смесь гауссиан с центрами в значениях
               проб и одинаковой шириной: l(x) = (1/n_good)·Σ N(x; xᵢ, h²) по хорошим, g(x) — так же по плохим.
               Ширина h = 0.1 × длина диапазона: **0.3** для `log_lr`, **0.05** для `subsample`.
            3. Совместные плотности — произведения по параметрам: l(x) = l_lr(x)·l_sub(x), g(x) = g_lr(x)·g_sub(x).
            4. Для каждого кандидата посчитайте `log_l = ln l(x)`, `log_g = ln g(x)`, `log_ratio = log_l − log_g`
               и `rank` — место по убыванию log_ratio (1 — следующая проба TPE).

            ## Подсказки

            - TPE ищет место, где хорошие пробы плотные, а плохие — редкие: максимум l(x)/g(x) эквивалентен максимуму EI.
            - Плотность одной смеси для всех кандидатов сразу: `norm.pdf(cand[:, None], loc=centers[None, :], scale=h).mean(axis=1)`.
            - Хорошие пробы: `df.nsmallest(6, 'loss')`, плохие — все остальные. Логарифм произведения — сумма логарифмов;
              rank: `(-log_ratio).argsort().argsort() + 1`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `candidate, log_l, log_g, log_ratio, rank` — 24 строки в порядке `candidates.csv`.
            Логарифмы — натуральные, округлить до 4 знаков.
        """,
        "solution": """
            from scipy.stats import norm

            N_GOOD = 6
            BANDWIDTH = {'log_lr': 0.3, 'subsample': 0.05}


            def parzen_logpdf(x, centers, h):
                # Смесь гауссиан с центрами в пробах: (1/n)·Σ N(x; xᵢ, h²)
                return np.log(norm.pdf(x[:, None], loc=centers[None, :], scale=h).mean(axis=1))


            def true_loss(log_lr, subsample):
                # «Истинный» ландшафт, из которого сгенерирована история (без шума) — только для сравнения
                return 0.20 + 0.04 * (log_lr + 1.2) ** 2 + 0.3 * (subsample - 0.8) ** 2


            def main():
                trials = pd.read_csv('input.csv')
                cand = pd.read_csv('candidates.csv')

                good = trials.nsmallest(N_GOOD, 'loss')
                bad = trials.drop(good.index)

                log_l = sum(parzen_logpdf(cand[p].to_numpy(), good[p].to_numpy(), h) for p, h in BANDWIDTH.items())
                log_g = sum(parzen_logpdf(cand[p].to_numpy(), bad[p].to_numpy(), h) for p, h in BANDWIDTH.items())
                log_ratio = log_l - log_g
                rank = (-log_ratio).argsort().argsort() + 1

                result = pd.DataFrame({'candidate': cand['candidate'], 'log_l': log_l, 'log_g': log_g,
                                       'log_ratio': log_ratio, 'rank': rank})

                chosen = cand.loc[rank == 1].iloc[0]
                random_pick = cand.iloc[0]
                print(f"TPE выбрал кандидата {int(chosen['candidate'])}: истинная ошибка "
                      f"{true_loss(chosen['log_lr'], chosen['subsample']):.4f}; "
                      f"лучшая в истории: {trials['loss'].min():.4f}; "
                      f"случайный (первый) кандидат: {true_loss(random_pick['log_lr'], random_pick['subsample']):.4f}")
                return result.round(4)
        """,
        "review": """
            **Идея.** TPE не моделирует саму функцию потерь, как GP. Он делит историю на хорошие и плохие пробы
            и для каждой группы оценивает, *где* в пространстве гиперпараметров они лежат. Хороший кандидат — там,
            где хорошие пробы плотные, а плохие — редкие: максимум l/g. Плотности строятся по каждому параметру
            отдельно, поэтому метод легко работает с целыми, категориальными и условными параметрами и с сотнями проб.

            **Типичные ошибки:**
            - Взяли хорошими пробы с наибольшим loss (перепутали направление) — rank перевернулся.
            - Использовали одну общую ширину для обоих параметров — у subsample диапазон в 6 раз меньше,
              и его плотность «расплывается».
            - Считали `norm.pdf(..., scale=h**2)` — в scipy `scale` — это стандартное отклонение, а не дисперсия.
            - Перемножили плотности, а потом взяли логарифм очень маленьких чисел без проблем — здесь можно, но в
              больших размерностях надёжнее складывать логарифмы.

            **Подробнее.** Настоящий TPE в Optuna подбирает ширину ядер адаптивно, добавляет априорную компоненту
            и сам генерирует кандидатов из l(x) (по умолчанию 24 штуки — как у нас). Hyperopt устроен так же,
            но пространство задаётся отдельно от objective, а функция всегда минимизируется.

            **Что вынести:** TPE = «где лежат хорошие пробы» / «где лежат плохие»; γ задаёт, сколько проб считать
            хорошими. → шпаргалка «Optuna / Hyperopt».
        """,
    },
    # ------------------------------------------------------------ Путь 3, этап 2
    {
        "name": "partial_dependence",
        "title": "Частичные зависимости вручную и взаимодействие признаков",
        "data": data_partial_dependence,
        "task": """
            Дан файл `input.csv`: признаки `x0, x1, x2` и целевая `y` (600 строк). Данные порождены формулой
            y = x0² + 2·x1 + x1·x2 + шум — влияние x1 зависит от x2.

            1. Обучите `HistGradientBoostingRegressor(max_iter=200, random_state=0)` на всех строках.
            2. Посчитайте **PDP вручную** для признаков `x0` и `x1` на сетке значений **−1.5, −0.75, 0, 0.75, 1.5**:
               для значения v скопируйте данные, во **всех** строках замените признак на v, предскажите и усредните —
               это `pd`.
            3. То же усреднение отдельно по строкам с x2 < 0 (`pd_x2_neg`) и с x2 ≥ 0 (`pd_x2_pos`) — модель обучена
               одна, меняется только набор строк, по которым усредняем.

            ## Подсказки

            - Частичная зависимость — среднее ICE-кривых: f(v, x_C⁽ⁱ⁾), усреднённое по объектам. Если кривые
              в подгруппах параллельны — взаимодействия нет; если наклоны разные — есть.
            - `X_mod = X.copy(); X_mod[:, j] = v; model.predict(X_mod)` — и это ICE для всех строк сразу;
              `pd` — `.mean()`, подгруппы — среднее по маске `X[:, 2] < 0`.
            - Сверьтесь с `sklearn.inspection.partial_dependence(model, X, [j], custom_values={j: grid}, method='brute')`.
              Наклон PDP по x1 в подгруппах должен быть ≈ 2 + E[x2 | подгруппа], т.е. около 1 и 3.

            ## Ожидаемый формат выходного файла

            CSV с колонками `feature, value, pd, pd_x2_neg, pd_x2_pos` — 10 строк: сначала 5 значений для `x0`,
            затем 5 для `x1`, значения по возрастанию. Числа округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.inspection import partial_dependence

            GRID = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
            FEATURES = ['x0', 'x1', 'x2']


            def main():
                df = pd.read_csv('input.csv')
                X, y = df[FEATURES].to_numpy(), df['y'].to_numpy()
                model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, y)
                neg = X[:, 2] < 0

                rows = []
                for j, name in [(0, 'x0'), (1, 'x1')]:
                    for v in GRID:
                        X_mod = X.copy()
                        X_mod[:, j] = v                         # «что если» для всех объектов сразу
                        ice = model.predict(X_mod)              # значения ICE-кривых в точке v
                        rows.append({'feature': name, 'value': v, 'pd': ice.mean(),
                                     'pd_x2_neg': ice[neg].mean(), 'pd_x2_pos': ice[~neg].mean()})

                    # Сверка с библиотекой
                    ref = partial_dependence(model, X, [j], custom_values={j: GRID}, method='brute')
                    assert np.allclose(ref['average'][0], [r['pd'] for r in rows[-5:]])

                result = pd.DataFrame(rows)
                for name in ['x0', 'x1']:
                    part = result[result['feature'] == name]
                    slope = lambda c: (part[c].iloc[-1] - part[c].iloc[0]) / 3
                    print(f"{name}: наклон от −1.5 до 1.5 — x2<0: {slope('pd_x2_neg'):.2f}, x2≥0: {slope('pd_x2_pos'):.2f}")
                return result.round(4)
        """,
        "review": """
            **Идея.** PDP отвечает на вопрос «как в среднем меняется прогноз, если всем объектам поставить x_S = v».
            По `x0` получается парабола, причём одинаковой формы в обеих подгруппах — `x0` ни с кем не взаимодействует,
            подгруппы отличаются лишь сдвигом. По `x1` — почти прямые, но с разным наклоном: ≈ 1 при x2 < 0
            и ≈ 3 при x2 ≥ 0. Общий PDP показал бы наклон ≈ 2 и спрятал бы это взаимодействие — именно поэтому
            смотрят на ICE-кривые и PDP по подгруппам.

            **Типичные ошибки:**
            - Усредняли y по объектам с x1 ≈ v вместо подстановки v во все строки — это условное среднее, а не PDP:
              в нём смешаны эффекты коррелированных признаков.
            - Изменили `X` на месте без копии — следующие точки сетки считаются на испорченных данных.
            - Обучили отдельные модели на подгруппах — нужен один и тот же f, меняется только набор усредняемых строк.
            - Сетку взяли по перцентилям (как sklearn по умолчанию), а не заданную в условии.

            **Что вынести:** PDP = среднее ICE; параллельные ICE — нет взаимодействий, веер — есть. PDP описывает
            реакцию *модели*, а не причинную связь, и ненадёжен при сильно коррелированных признаках (тогда — ALE).
            → шпаргалка «Частичные зависимости».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям по классическому ML:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=701):
        build(exercise, exercise.get("seed", seed))


if __name__ == "__main__":
    main()
