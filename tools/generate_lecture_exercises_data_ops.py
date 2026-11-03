#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор практических упражнений к лекциям по инструментам данных, MLOps, production ML
и надёжному/объяснимому ML.

Темы: numpy_ml, pandas_ml, matplotlib_seaborn_ml, cicd_ml, feature_stores, mlops_best_practices,
model_monitoring_ab_testing, model_serialization, production_ml_best_practices, fairness_ml,
explainable_ai_xai, lime, multi_armed_bandits, adversarial_robustness.

Инженерные темы превращены в вычислительные задачи: quality gate по таблице запусков,
point-in-time join, хеш версии данных, PSI и z-тест, проверка сериализованной модели,
перцентили задержки и SLA и т.д.

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_data_ops.py
"""

import hashlib
import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build, split_column

# ---------------------------------------------------------------- данные


def data_numpy(rng):
    n = 30
    height = rng.normal(172, 9, n).round(1)
    weight = (0.9 * (height - 172) + 72 + rng.normal(0, 7, n)).round(1)
    age = rng.integers(18, 65, n).astype(float)
    height[7], weight[19], age[25] = 205.0, 128.0, 89.0            # по одному «экстремальному» значению
    df = pd.DataFrame({"id": np.arange(101, 101 + n), "height": height, "weight": weight, "age": age})
    weights = {"w": [0.8, -0.5, 0.3], "b": 0.1}
    return {"input.csv": df, "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_pandas(rng):
    n = 25
    cities = rng.choice(["Москва", "Казань", "Самара"], n)
    age = rng.integers(18, 70, n).astype(float)
    age[rng.choice(n, 6, replace=False)] = np.nan
    customers = pd.DataFrame({"customer_id": np.arange(1, n + 1), "city": cities, "age": age})

    rows = []
    for cid in range(1, n + 1):
        if cid in (4, 13, 21):                                     # клиенты без заказов
            continue
        for _ in range(rng.integers(1, 6)):
            day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 90)))
            rows.append({"customer_id": cid, "order_date": day.strftime("%Y-%m-%d"),
                         "amount": f"{rng.gamma(2.0, 600):.2f}"})
    orders = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1001, 1001 + len(orders)))
    comma = rng.choice(len(orders), 6, replace=False)
    orders.loc[comma, "amount"] = orders.loc[comma, "amount"].str.replace(".", ",", regex=False)
    bad = [i for i in rng.choice(len(orders), 6, replace=False) if i not in comma][:3]
    orders.loc[bad, "amount"] = "—"
    dups = orders.iloc[[2, 10, 17]]                                # дубли выгрузки
    orders = pd.concat([orders, dups]).reset_index(drop=True)
    return {"input.csv": customers, "orders.csv": orders}


def data_matplotlib(rng):
    a = rng.normal(50, 8, 48).round(1)
    b = rng.lognormal(3.4, 0.45, 55).round(1)
    c = np.concatenate([rng.normal(40, 5, 40), [8.0, 75.5]]).round(1)
    df = pd.DataFrame({"group": ["A"] * len(a) + ["B"] * len(b) + ["C"] * len(c),
                       "value": np.concatenate([a, b, c])})
    df = df.sample(frac=1, random_state=3).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df


def data_cicd(rng):
    # run_id, segment, tp, fp, fn; tn = N_segment − fp. Валидация: north P=120, N=380; south P=80, N=220
    counts = {
        "prod": [(90, 30, 30), (56, 24, 24)],
        "r1": [(96, 28, 24), (58, 22, 22)],
        "r2": [(100, 26, 20), (57, 23, 23)],
        "r3": [(104, 20, 16), (50, 30, 30)],
        "r4": [(91, 30, 29), (56, 24, 24)],
        "r5": [(105, 18, 15), (62, 18, 18)],
        "r6": [(95, 25, 20), (58, 22, 22)],
    }
    neg = {"north": 380, "south": 220}
    rows = []
    for run, segs in counts.items():
        for seg, (tp, fp, fn) in zip(["north", "south"], segs):
            rows.append({"run_id": run, "segment": seg, "tp": tp, "fp": fp, "fn": fn, "tn": neg[seg] - fp})
    meta = pd.DataFrame([
        {"run_id": "prod", "smoke_test": "pass", "latency_p95_ms": 95, "size_mb": 38},
        {"run_id": "r1", "smoke_test": "pass", "latency_p95_ms": 88, "size_mb": 35},
        {"run_id": "r2", "smoke_test": "pass", "latency_p95_ms": 118, "size_mb": 45},
        {"run_id": "r3", "smoke_test": "pass", "latency_p95_ms": 101, "size_mb": 40},
        {"run_id": "r4", "smoke_test": "pass", "latency_p95_ms": 90, "size_mb": 36},
        {"run_id": "r5", "smoke_test": "fail", "latency_p95_ms": 150, "size_mb": 44},
        {"run_id": "r6", "smoke_test": "pass", "latency_p95_ms": 97, "size_mb": 80},
    ])
    return {"input.csv": pd.DataFrame(rows), "meta.csv": meta}


def data_feature_stores(rng):
    start = pd.Timestamp("2026-01-01")
    orders, tickets = [], []
    for user in range(1, 7):
        days = np.sort(rng.choice(np.arange(0, 80), 5, replace=False))
        if user == 5:
            days = np.array([2, 6, 11])                            # давно не обновлялся: сработает TTL
        level = int(rng.integers(1, 4))
        for d in days:
            level += int(rng.integers(0, 3))
            orders.append({"user_id": user, "ts": start + pd.Timedelta(days=int(d)), "orders_30d": level})
        for d in np.sort(rng.choice(np.arange(5, 85), 3, replace=False)):
            tickets.append({"user_id": user, "ts": start + pd.Timedelta(days=int(d)), "tickets_90d": int(rng.integers(0, 6))})
    orders, tickets = pd.DataFrame(orders), pd.DataFrame(tickets)

    labels = []
    for i in range(16):
        user = int(rng.integers(1, 7))
        day = start + pd.Timedelta(days=int(rng.integers(20, 85)))
        labels.append({"user_id": user, "ts": day, "churned": int(rng.integers(0, 2))})
    labels = pd.DataFrame(labels)
    # метки ровно в день обновления признака: значение этого дня ещё недоступно
    for k, (u, j) in enumerate([(1, 3), (4, 2), (2, 3)]):
        labels.loc[k * 5, ["user_id", "ts"]] = [u, orders[orders["user_id"] == u]["ts"].iloc[j]]
    labels.loc[15, ["user_id", "ts"]] = [2, start + pd.Timedelta(days=1)]   # раньше первого значения
    labels.insert(0, "label_id", np.arange(1, len(labels) + 1))
    fmt = lambda df: df.assign(ts=df["ts"].dt.strftime("%Y-%m-%d"))
    return {"input.csv": fmt(labels),
            "orders_features.csv": fmt(orders.sample(frac=1, random_state=5).reset_index(drop=True)),
            "tickets_features.csv": fmt(tickets.sample(frac=1, random_state=6).reset_index(drop=True))}


def _data_version(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def data_mlops(rng):
    n = 60
    df = pd.DataFrame({"x1": rng.normal(0, 1, n).round(3), "x2": rng.normal(0, 1, n).round(3)})
    df["y"] = (df["x1"] + 0.5 * df["x2"] + rng.normal(0, 0.7, n) > 0).astype(int)
    current = df.to_csv(index=False, lineterminator="\n")          # build() сам запишет \n как \r\n на Windows
    old = df.copy()
    old.loc[17, "y"] = 1 - old.loc[17, "y"]                        # старая версия: одна метка была другой
    v_cur, v_old = _data_version(current), _data_version(old.to_csv(index=False, lineterminator="\n"))
    assert v_cur != v_old
    runs = [
        ("run_01", "FINISHED", v_old, {"max_depth": 3}, 0.8712, 0.012, 0.866),
        ("run_02", "FINISHED", v_old, {"max_depth": 6}, 0.8955, 0.020, 0.884),
        ("run_03", "FINISHED", v_cur, {"max_depth": 3}, 0.8790, 0.011, 0.871),
        ("run_04", "FAILED", v_cur, {"max_depth": 12}, 0.9010, 0.031, 0.000),
        ("run_05", "FINISHED", v_cur, {"max_depth": 4}, 0.8846, 0.015, 0.893),
        ("run_06", "FINISHED", v_cur, {"max_depth": 5}, 0.8861, 0.018, 0.879),
        ("run_07", "FINISHED", v_cur, {"max_depth": 5, "min_samples_leaf": 5}, 0.8861, 0.009, 0.880),
        ("run_08", "FINISHED", v_old, {"max_depth": 4}, 0.8803, 0.014, 0.875),
        ("run_09", "FINISHED", v_cur, {"max_depth": 8}, 0.8731, 0.024, 0.902),
    ]
    lines = []
    for i, (rid, status, ver, params, cv, std, test) in enumerate(runs):
        rec = {"run_id": rid, "time": f"2026-10-{i + 1:02d} 12:00", "status": status, "params": params,
               "metrics": {"cv_auc": cv, "cv_auc_std": std, "test_auc": test},
               "data_version": ver, "git_commit": hashlib.sha1(rid.encode()).hexdigest()[:7]}
        lines.append(json.dumps(rec, ensure_ascii=False))
    return {"input.csv": current, "runs.jsonl": "\n".join(lines) + "\n"}


def data_monitoring(rng):
    ref = pd.DataFrame({"sample": "reference", "income": rng.lognormal(10.8, 0.4, 3000),
                        "session_min": rng.gamma(3.0, 4.0, 3000), "basket": rng.normal(2500, 600, 3000)})
    cur = pd.DataFrame({"sample": "current", "income": rng.lognormal(10.83, 0.42, 1500),
                        "session_min": rng.gamma(3.0, 5.2, 1500), "basket": rng.normal(2480, 610, 1500)})
    df = pd.concat([ref, cur], ignore_index=True)
    df["income"] = df["income"].round(0)
    df["session_min"] = df["session_min"].round(2)
    df["basket"] = df["basket"].round(1)
    n = 5000
    ab = pd.DataFrame({"user_id": np.arange(1, 2 * n + 1), "group": rng.permutation(np.repeat(["A", "B"], n))})
    ab["converted"] = rng.binomial(1, np.where(ab["group"] == "A", 0.100, 0.113))
    return {"input.csv": df, "ab.csv": ab}


def data_serialization(rng):
    n = 300
    age = rng.integers(18, 70, n)
    income = rng.normal(60000, 18000, n).round(0)
    tenure = rng.integers(0, 120, n)
    n_products = rng.integers(1, 6, n)
    logit = 0.04 * (age - 40) - 0.00006 * (income - 60000) - 0.02 * (tenure - 60) + 0.5 * (n_products - 3)
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"id": np.arange(1, n + 1), "age": age, "income": income, "tenure": tenure,
                       "n_products": n_products, "target": target, "split": split_column(n, 0.3, rng)})
    feats = "age|income|tenure|n_products"
    registry = pd.DataFrame([
        {"model_id": "m1", "sklearn_version": "1.9.0", "numpy_version": "2.3.1", "features": feats},
        {"model_id": "m2", "sklearn_version": "1.8.2", "numpy_version": "2.5.4", "features": feats},
        {"model_id": "m3", "sklearn_version": "1.9.1", "numpy_version": "1.26.4", "features": feats},
        {"model_id": "m4", "sklearn_version": "1.9.1", "numpy_version": "2.5.4", "features": "age|tenure|income|n_products"},
        {"model_id": "m5", "sklearn_version": "1.9.1", "numpy_version": "2.5.4", "features": "age|income|tenure"},
        {"model_id": "m6", "sklearn_version": "1.19.0", "numpy_version": "2.5.4", "features": feats},
    ])
    env = {"python": "3.12", "sklearn_version": "1.9.1", "numpy_version": "2.5.4",
           "features": ["age", "income", "tenure", "n_products"]}
    return {"input.csv": df, "registry.csv": registry, "service_env.json": json.dumps(env, indent=2) + "\n"}


def data_production(rng):
    rows = []
    specs = {"v1": (3000, 62, 0.35, 0.004, 0.003), "v2": (3000, 48, 0.55, 0.009, 0.006)}
    rid = 1
    for ver, (n, median, sigma, p_timeout, p_error) in specs.items():
        lat = rng.lognormal(np.log(median), sigma, n)
        u = rng.uniform(size=n)
        status = np.where(u < p_timeout, "timeout", np.where(u < p_timeout + p_error, "error", "ok"))
        lat = np.where(status == "timeout", 300.0, np.where(status == "error", rng.uniform(2, 8, n), np.minimum(lat, 299.0)))
        for s, l in zip(status, lat):
            rows.append({"model_version": ver, "status": s, "latency_ms": round(float(l), 1)})
    df = pd.DataFrame(rows).sample(frac=1, random_state=11).reset_index(drop=True)
    df.insert(0, "request_id", np.arange(1, len(df) + 1))
    return df


def data_fairness(rng):
    n = 800
    group = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.5, n)
    logit = 1.8 * (2 * y - 1) + 0.9 * group - 0.45 + rng.normal(0, 1.3, n)   # скоры у группы 1 завышены
    score = (1 / (1 + np.exp(-logit))).round(3)
    return pd.DataFrame({"id": np.arange(1, n + 1), "group": group, "y_true": y, "score": score})


def data_xai(rng):
    n = 400
    latent = rng.normal(0, 1, n)                                   # скрытая величина
    x2 = rng.uniform(-2, 2, n)
    x1, x3 = latent + rng.normal(0, 0.35, n), latent + rng.normal(0, 0.35, n)   # два её измерения
    x4 = rng.normal(0, 1, n)                                       # шум
    y = 3 * latent + 1.5 * np.sin(2 * x2) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "x1": x1.round(3), "x2": x2.round(3),
                       "x3": x3.round(3), "x4": x4.round(3), "y": y.round(3)})
    df["split"] = split_column(n, 0.25, rng)
    return df


def data_lime(rng):
    points = pd.DataFrame({"id": [1, 2, 3], "x1": [1.2, -1.2, 0.1], "x2": [0.0, 0.3, -0.4], "x3": [0.5, 0.5, 2.0]})
    noise = pd.DataFrame(rng.normal(0, 1, (1000, 3)).round(4), columns=["e1", "e2", "e3"])
    return {"input.csv": points, "noise.csv": noise}


def data_bandits(rng):
    T = 3000
    p = np.array([0.10, 0.15, 0.22, 0.18])
    df = pd.DataFrame({"t": np.arange(1, T + 1), "u": rng.uniform(size=T).round(6),
                       "rand_arm": rng.integers(0, 4, T)})
    for k in range(4):
        df[f"r{k}"] = rng.binomial(1, p[k], T)
    return df


def data_adversarial(rng):
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    mask = (y == 1) | (y == 7)
    X, y = X[mask] / 16.0, (y[mask] == 7).astype(int)
    df = pd.DataFrame(X.round(4), columns=[f"p{j}" for j in range(64)])
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["label"] = y
    df["split"] = split_column(len(df), 0.3, rng)
    return df


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    # ------------------------------------------------------------ numpy_ml
    {
        "name": "numpy_ml",
        "title": "NumPy: оси, broadcasting, маски и матричное умножение",
        "data": data_numpy,
        "task": """
            Дан файл `input.csv`: 30 человек с признаками `height` (см), `weight` (кг), `age` (лет),
            и файл `weights.json` с весами линейной модели `w` (3 числа в порядке height, weight, age) и сдвигом `b`.
            Решите задачу **без циклов по объектам** — только операциями над массивами NumPy.

            1. Соберите матрицу X размера 30 × 3 и стандартизируйте её по столбцам:
               Z = (X − mean) / std, где mean и std считаются по оси объектов (`axis=0`), std — с `ddof=0`.
            2. `score` — прогноз линейной модели на стандартизованных признаках: score = Z·w + b (оператор `@`).
            3. `n_extreme` — сколько признаков у объекта по модулю дальше двух стандартных отклонений: число j с |Zᵢⱼ| > 2.
            4. `nearest_id` — `id` ближайшего **другого** человека по евклидову расстоянию в пространстве Z,
               `nearest_dist` — это расстояние. Матрицу расстояний 30 × 30 постройте через broadcasting;
               расстояние объекта до самого себя исключите (например, поставьте на диагональ `np.inf`).
               Ничьих в данных нет.

            ## Подсказки

            - Проверьте формы: `X.mean(axis=0)` — вектор из 3 чисел, и `X - mean` вычитает его из каждой строки
              (формы (30, 3) и (3,) совместимы по правилу broadcasting).
            - `n_extreme` — это булева маска `np.abs(Z) > 2`, просуммированная по оси признаков (`axis=1`).
            - Разности всех пар: `D = Z[:, None, :] - Z[None, :, :]` (форма 30 × 30 × 3), затем
              `dist = np.sqrt((D ** 2).sum(axis=2))`, `np.fill_diagonal(dist, np.inf)`, `j = dist.argmin(axis=1)`,
              и `nearest_id = ids[j]` — индексация массивом.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, score, n_extreme, nearest_id, nearest_dist` — 30 строк в исходном порядке;
            `score` и `nearest_dist` округлить до 4 знаков, `n_extreme` и `nearest_id` — целые.
        """,
        "solution": """
            import json

            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler


            def main():
                df = pd.read_csv('input.csv')
                params = json.load(open('weights.json', encoding='utf-8'))
                ids = df['id'].to_numpy()
                X = df[['height', 'weight', 'age']].to_numpy(dtype=float)     # (30, 3)
                w, b = np.array(params['w']), params['b']

                # 1) Стандартизация: агрегаты по оси объектов, broadcasting (30, 3) − (3,)
                Z = (X - X.mean(axis=0)) / X.std(axis=0)
                assert np.allclose(Z, StandardScaler().fit_transform(X))

                # 2) Линейная модель для всех объектов сразу
                score = Z @ w + b                                             # (30,)

                # 3) Булева маска и сумма по оси признаков
                n_extreme = (np.abs(Z) > 2).sum(axis=1)

                # 4) Попарные расстояния через broadcasting: (30, 1, 3) − (1, 30, 3) → (30, 30, 3)
                dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
                np.fill_diagonal(dist, np.inf)                                # себя не считаем соседом
                j = dist.argmin(axis=1)
                nearest_dist = dist[np.arange(len(Z)), j]
                second = np.sort(dist, axis=1)[:, 1]
                assert np.all(second - nearest_dist > 1e-9)                   # ничьих нет

                # Сверка с sklearn: первый сосед — сам объект, второй — ближайший другой
                nn_dist, nn_idx = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
                assert np.array_equal(nn_idx[:, 1], j) and np.allclose(nn_dist[:, 1], nearest_dist)

                print('объекты с экстремальными значениями:', ids[n_extreme > 0])
                return pd.DataFrame({'id': ids, 'score': score.round(4), 'n_extreme': n_extreme,
                                     'nearest_id': ids[j], 'nearest_dist': nearest_dist.round(4)})
        """,
        "review": """
            **Идея.** Весь расчёт — четыре строки над массивами: агрегаты по `axis=0`, broadcasting при вычитании вектора
            средних, матричное умножение `Z @ w` вместо цикла и булева маска с суммой по `axis=1`. Матрица попарных
            расстояний получается добавлением осей `None`: формы (30, 1, 3) и (1, 30, 3) растягиваются до (30, 30, 3).

            **Типичные ошибки:**
            - `X.mean()` без `axis` — одно среднее на всю матрицу: стандартизация смешивает рост, вес и возраст.
            - `X.std(ddof=1)` (как в pandas `df.std()`) — все Z в √(30/29) раза меньше, `score` и расстояния другие.
            - Забыли исключить диагональ — ближайшим «соседом» каждого объекта оказывается он сам с расстоянием 0.
            - `Z * w` вместо `Z @ w` — получается матрица 30 × 3 поэлементных произведений, а не прогноз.
            - Вернули индекс строки `j` вместо `id[j]` — `nearest_id` сдвинут на 101.

            **Что вынести:** «axis — ось, которая схлопывается», broadcasting сравнивает формы справа налево, а добавление
            оси `None` — главный приём для попарных вычислений (расстояния в k-NN, k-means). → шпаргалка «NumPy для ML».
        """,
    },
    # ------------------------------------------------------------ pandas_ml
    {
        "name": "pandas_ml",
        "title": "Pandas: чистка, пропуски по группам, groupby и merge",
        "data": data_pandas,
        "task": """
            Дано две таблицы:

            - `input.csv` — клиенты: `customer_id`, `city`, `age` (есть пропуски);
            - `orders.csv` — заказы: `order_id`, `customer_id`, `order_date`, `amount`.

            Выгрузка заказов «грязная»: `amount` прочитается как текст — в нём встречаются суммы с десятичной запятой
            (`1234,56`) и прочерки `—`; кроме того, несколько строк продублированы целиком.
            Постройте по каждому клиенту таблицу признаков:

            1. Удалите полностью повторяющиеся строки заказов (`drop_duplicates`).
            2. Приведите `amount` к числу: замените запятую на точку и примените `pd.to_numeric(..., errors='coerce')`;
               заказы, где сумма так и не стала числом, удалите.
            3. Заполните пропуски `age` **медианой возраста клиентов того же города** (медиана — по известным значениям).
            4. По каждому клиенту: `n_orders` — число заказов, `total_amount` — сумма, `avg_amount` — средний чек,
               `days_since_last` — сколько дней прошло от последнего заказа до даты отчёта **2026-04-01**.
            5. Присоедините агрегаты к клиентам так, чтобы клиенты **без заказов не потерялись** (left join).
               Для них `n_orders = 0`, `total_amount = 0`, `avg_amount = 0`, `days_since_last = -1`.

            ## Подсказки

            - Начните с осмотра: `orders.info()` покажет, что `amount` имеет тип `object`, а `orders.duplicated().sum()` —
              сколько дублей.
            - Медиана по группе, «растянутая» обратно на строки: `df.groupby('city')['age'].transform('median')` —
              ей и заполняйте `fillna`.
            - Агрегаты: `orders.groupby('customer_id').agg(n_orders=('order_id', 'size'), total_amount=('amount', 'sum'), ...)`,
              затем `customers.merge(agg, on='customer_id', how='left')` и `fillna` для клиентов без заказов.
              Разницу дат получите как `(pd.Timestamp('2026-04-01') - last).dt.days`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `customer_id, city, age, n_orders, total_amount, avg_amount, days_since_last` —
            25 строк по возрастанию `customer_id`; `age`, `total_amount`, `avg_amount` округлить до 2 знаков,
            `n_orders` и `days_since_last` — целые.
        """,
        "solution": """
            REPORT_DATE = pd.Timestamp('2026-04-01')


            def main():
                customers = pd.read_csv('input.csv')
                orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
                print(orders.dtypes['amount'], '| дублей:', orders.duplicated().sum())

                # 1) Дубли выгрузки
                orders = orders.drop_duplicates()

                # 2) Сумма: десятичная запятая → точка, мусор → NaN → удаляем
                orders['amount'] = pd.to_numeric(orders['amount'].str.replace(',', '.', regex=False), errors='coerce')
                print('нечисловых сумм:', orders['amount'].isna().sum())
                orders = orders.dropna(subset=['amount'])

                # 3) Пропуски возраста — медианой своего города
                city_median = customers.groupby('city')['age'].transform('median')
                customers['age'] = customers['age'].fillna(city_median)

                # 4) Агрегаты по клиенту
                agg = orders.groupby('customer_id').agg(
                    n_orders=('order_id', 'size'),
                    total_amount=('amount', 'sum'),
                    avg_amount=('amount', 'mean'),
                    last_order=('order_date', 'max'),
                ).reset_index()
                agg['days_since_last'] = (REPORT_DATE - agg['last_order']).dt.days

                # 5) Left join: клиенты без заказов остаются
                result = customers.merge(agg.drop(columns='last_order'), on='customer_id', how='left')
                assert len(result) == len(customers)
                result = result.fillna({'n_orders': 0, 'total_amount': 0, 'avg_amount': 0, 'days_since_last': -1})
                result[['n_orders', 'days_since_last']] = result[['n_orders', 'days_since_last']].astype(int)

                print(result.groupby('city')[['total_amount']].mean().round(1))
                result = result.sort_values('customer_id').reset_index(drop=True)
                return result.round({'age': 2, 'total_amount': 2, 'avg_amount': 2})
        """,
        "review": """
            **Идея.** Типичная подготовка табличных данных: осмотр (`info`, `duplicated`), приведение типов, заполнение
            пропусков по группе через `groupby(...).transform`, агрегаты `groupby().agg()` с именованными столбцами и
            соединение `merge(how='left')`, после которого «отсутствие заказов» явно превращается в нули.

            **Типичные ошибки:**
            - Не удалили дубли — у клиентов 3, 5 и 8 лишний заказ: `n_orders` и суммы завышены.
            - `pd.to_numeric(errors='coerce')` без замены запятой — суммы `1234,56` превращаются в NaN и теряются.
            - Заполнили возраст общей медианой (или средним по городу) вместо медианы города.
            - `merge` по умолчанию (`how='inner'`) — клиенты 4, 13, 21 пропадают, строк 22 вместо 25.
            - Забыли `fillna` после left join — у клиентов без заказов NaN вместо 0 и −1.

            **Что вынести:** первые команды на любом датасете — `info()`, `isna().sum()`, `duplicated().sum()`;
            `transform` возвращает результат по группе той же длины, что и таблица, а после left join всегда
            проверяйте число строк и пропуски. → шпаргалка «Pandas для ML».
        """,
    },
    # ------------------------------------------------------------ matplotlib_seaborn_ml
    {
        "name": "matplotlib_seaborn_ml",
        "title": "Что показывают ящик с усами и гистограмма: считаем вручную",
        "data": data_matplotlib,
        "task": """
            Дан файл `input.csv`: значения `value` для трёх групп `A`, `B`, `C`. Прежде чем строить
            `sns.boxplot(x='group', y='value')` и гистограммы, посчитайте **вручную** числа, которые эти графики покажут.

            Для каждой группы:

            1. `n` — число объектов; `median`, `q1`, `q3` — медиана и квартили через `np.percentile`
               (линейная интерполяция — по умолчанию; так считают и matplotlib, и seaborn).
            2. Усы ящика (правило Тьюки, `whis = 1.5`): IQR = Q3 − Q1, границы Q1 − 1.5·IQR и Q3 + 1.5·IQR.
               `whisker_low` — **наименьшее значение данных**, которое не меньше нижней границы;
               `whisker_high` — **наибольшее значение данных**, не больше верхней. Усы заканчиваются на реальной точке,
               а не на самой границе!
            3. `n_outliers` — число точек за усами (их boxplot рисует отдельными кружками).
            4. Гистограмма группы с **10 корзинами** одинаковой ширины от min до max (как `np.histogram(values, bins=10)`
               и `ax.hist(values, bins=10)`): `mode_bin_left` — левая граница самой заполненной корзины,
               `mode_bin_count` — сколько в ней точек (при равенстве — левая из корзин, как `argmax`).

            ## Подсказки

            - Начните с `df.groupby('group')['value']` и функции, которая по массиву значений возвращает словарь статистик.
            - Усы: `v[v >= q1 - 1.5 * iqr].min()` и `v[v <= q3 + 1.5 * iqr].max()`; выбросы — всё, что строго левее
              нижнего уса или правее верхнего.
            - `counts, edges = np.histogram(v, bins=10)`, `k = counts.argmax()` → `edges[k]`, `counts[k]`.
              Сверьте с `matplotlib.cbook.boxplot_stats(v)` (ключи `q1`, `med`, `q3`, `whislo`, `whishi`, `fliers`).

            ## Ожидаемый формат выходного файла

            CSV с колонками `group, n, median, q1, q3, whisker_low, whisker_high, n_outliers, mode_bin_left, mode_bin_count` —
            3 строки в порядке `A, B, C`; дробные значения округлить до 4 знаков.
        """,
        "solution": """
            import matplotlib
            matplotlib.use('Agg')                       # рисуем без окна
            import matplotlib.pyplot as plt
            from matplotlib import cbook


            def box_and_hist(v):
                q1, median, q3 = np.percentile(v, [25, 50, 75])
                iqr = q3 - q1
                lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                whisker_low = v[v >= lo_fence].min()      # ус заканчивается на реальной точке
                whisker_high = v[v <= hi_fence].max()
                n_outliers = int(((v < whisker_low) | (v > whisker_high)).sum())
                counts, edges = np.histogram(v, bins=10)
                k = counts.argmax()
                return {'n': len(v), 'median': median, 'q1': q1, 'q3': q3, 'whisker_low': whisker_low,
                        'whisker_high': whisker_high, 'n_outliers': n_outliers,
                        'mode_bin_left': edges[k], 'mode_bin_count': int(counts[k])}


            def main():
                df = pd.read_csv('input.csv')
                rows = []
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for g, part in df.groupby('group'):
                    v = part['value'].to_numpy()
                    s = box_and_hist(v)

                    # Сверка с тем, что нарисует matplotlib
                    ref = cbook.boxplot_stats(v)[0]
                    assert np.isclose(ref['q1'], s['q1']) and np.isclose(ref['q3'], s['q3'])
                    assert np.isclose(ref['whislo'], s['whisker_low']) and np.isclose(ref['whishi'], s['whisker_high'])
                    assert len(ref['fliers']) == s['n_outliers']
                    counts, _, _ = axes[1].hist(v, bins=10, alpha=0.5, label=g)
                    assert counts.max() == s['mode_bin_count']

                    rows.append({'group': g, **s})
                axes[0].boxplot([df.loc[df['group'] == g, 'value'] for g in 'ABC'], tick_labels=list('ABC'))
                axes[1].legend()
                plt.close(fig)

                result = pd.DataFrame(rows)
                print(result.round(2).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** Ящик с усами — это пять чисел: Q1, медиана, Q3 и концы усов, плюс отдельные точки-выбросы.
            Граница Q ± 1.5·IQR задаёт лишь «забор»: сам ус рисуется до последней **реальной** точки внутри забора.
            Гистограмма — подсчёт точек в равных корзинах от min до max, поэтому два дальних выброса (группа C)
            растягивает диапазон, корзины становятся шире, и форма распределения «сплющивается».

            **Типичные ошибки:**
            - Записали в `whisker_low/high` саму границу Q1 − 1.5·IQR — значения не совпадают с данными.
            - Квартили через `df.quantile` с другой интерполяцией или «медиана половинок» — Q1/Q3 чуть другие.
            - Посчитали гистограмму по всем группам сразу или с `range` по общей шкале — корзины другие.
            - В группе B (скошенное логнормальное распределение) выбросы только справа — это не ошибка данных,
              а свойство распределения: не спешите их удалять.

            **Что вынести:** прежде чем делать выводы по графику, полезно понимать, какие числа он кодирует; при
            выбросах гистограмму стоит строить с ограниченным диапазоном или в логарифмической шкале.
            → шпаргалка «Matplotlib/Seaborn для ML визуализации».
        """,
    },
    # ------------------------------------------------------------ cicd_ml
    {
        "name": "cicd_ml",
        "title": "Quality gate в CI/CD: пропустить модель или остановить конвейер",
        "data": data_cicd,
        "task": """
            CI-конвейер обучил шесть моделей-кандидатов `r1…r6`. Все они (и текущая продакшен-модель `prod`)
            должны оцениваться на **одной и той же** зафиксированной валидационной выборке из двух сегментов.

            - `input.csv` — матрицы ошибок по сегментам: `run_id, segment, tp, fp, fn, tn`;
            - `meta.csv` — результаты остальных шагов: `smoke_test` (`pass`/`fail`), `latency_p95_ms`, `size_mb`.

            Посчитайте F1 и примените quality gate. Для каждого кандидата:

            1. **F1 по сегменту** = 2·TP / (2·TP + FP + FN). **Общий F1** — по суммарным TP, FP, FN обоих сегментов.
            2. Бейзлайн — константная модель «всегда 1» на той же выборке `prod`: TP = P, FP = N, FN = 0,
               где P = TP + FN и N = FP + TN, суммарно по сегментам.
            3. Проверки (в таком порядке и с такими именами):
               - `smoke` — `smoke_test == 'pass'`;
               - `same_data` — в **каждом** сегменте у кандидата те же P и N, что у `prod`;
               - `vs_prod` — общий F1 ≥ F1(`prod`) + 0.005;
               - `vs_baseline` — общий F1 ≥ F1(бейзлайна) + 0.02;
               - `segments` — в каждом сегменте F1 ≥ F1 этого сегмента у `prod` − 0.01;
               - `latency` — `latency_p95_ms` ≤ 120;
               - `size` — `size_mb` ≤ 50.
            4. `decision` = `PASS`, если пройдены все проверки, иначе `FAIL`; `failed` — имена непройденных проверок
               через `;` в порядке списка выше (или `none`).
            5. `promote` = 1 у **одного** прошедшего кандидата с наибольшим общим F1, у остальных 0.

            ## Подсказки

            - Удобно сначала развернуть таблицу: `pivot` по сегментам или `groupby('run_id')[['tp','fp','fn']].sum()` для общего F1.
            - Проверка `same_data` ловит кандидата, оценённого на другой выборке: его метрика несравнима с `prod`,
              даже если выглядит хорошо.
            - Соберите проверки в словарь `{имя: bool}` (Python сохраняет порядок ключей) — тогда
              `failed = ';'.join(k for k, ok in checks.items() if not ok) or 'none'`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `run_id, f1, delta_vs_prod, decision, failed, promote` — 6 строк в порядке `r1…r6`;
            `delta_vs_prod` = F1 кандидата − F1(`prod`); `f1` и `delta_vs_prod` округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import f1_score

            MIN_GAIN_PROD, MIN_GAIN_BASE, MAX_SEG_DROP = 0.005, 0.02, 0.01
            MAX_LATENCY, MAX_SIZE = 120, 50


            def f1(tp, fp, fn):
                return 2 * tp / (2 * tp + fp + fn)


            def main():
                cm = pd.read_csv('input.csv')
                meta = pd.read_csv('meta.csv').set_index('run_id')
                cm['P'], cm['N'] = cm['tp'] + cm['fn'], cm['fp'] + cm['tn']
                cm['f1'] = f1(cm['tp'], cm['fp'], cm['fn'])
                seg = cm.set_index(['run_id', 'segment'])
                total = cm.groupby('run_id')[['tp', 'fp', 'fn', 'P', 'N']].sum()
                total['f1'] = f1(total['tp'], total['fp'], total['fn'])

                prod_f1 = total.loc['prod', 'f1']
                P, N = total.loc['prod', 'P'], total.loc['prod', 'N']
                base_f1 = f1(P, N, 0)                          # «всегда 1»: TP = P, FP = N, FN = 0
                # та же формула через sklearn
                y = np.r_[np.ones(P), np.zeros(N)]
                assert np.isclose(base_f1, f1_score(y, np.ones_like(y)))
                print(f"F1 prod = {prod_f1:.4f}, F1 бейзлайна = {base_f1:.4f}")

                rows = []
                for run in [r for r in total.index if r != 'prod']:
                    s, p = seg.loc[run], seg.loc['prod']
                    checks = {
                        'smoke': meta.loc[run, 'smoke_test'] == 'pass',
                        'same_data': bool(((s['P'] == p['P']) & (s['N'] == p['N'])).all()),
                        'vs_prod': total.loc[run, 'f1'] >= prod_f1 + MIN_GAIN_PROD,
                        'vs_baseline': total.loc[run, 'f1'] >= base_f1 + MIN_GAIN_BASE,
                        'segments': bool((s['f1'] >= p['f1'] - MAX_SEG_DROP).all()),
                        'latency': meta.loc[run, 'latency_p95_ms'] <= MAX_LATENCY,
                        'size': meta.loc[run, 'size_mb'] <= MAX_SIZE,
                    }
                    failed = ';'.join(k for k, ok in checks.items() if not ok) or 'none'
                    rows.append({'run_id': run, 'f1': total.loc[run, 'f1'],
                                 'delta_vs_prod': total.loc[run, 'f1'] - prod_f1,
                                 'decision': 'PASS' if failed == 'none' else 'FAIL', 'failed': failed})

                result = pd.DataFrame(rows)
                passed = result[result['decision'] == 'PASS']
                result['promote'] = 0
                if len(passed):
                    result.loc[passed['f1'].idxmax(), 'promote'] = 1
                print(result.to_string(index=False))
                return result.round({'f1': 4, 'delta_vs_prod': 4})
        """,
        "review": """
            **Идея.** Quality gate — набор автоматических правил, каждое из которых может «закрыть шлагбаум»:
            работоспособность (smoke-тест), сравнимость (та же выборка), качество относительно продакшена и бейзлайна,
            отсутствие деградации в сегментах и инженерные ограничения. В CI скрипт гейта при `FAIL` завершается
            с ненулевым кодом выхода, и конвейер останавливается.

            **Типичные ошибки:**
            - Общий F1 как среднее F1 сегментов (macro) вместо F1 по суммарным счётчикам — у r3 разница заметна.
            - Не проверили `same_data` — r6 выглядит лучше `prod`, но оценён на другой выборке (в north P = 115).
            - Проверили только общий F1: r3 улучшает метрику в целом за счёт провала в сегменте south.
            - Строгое `>` вместо `≥` в порогах или запас 0.5 п.п. прибавлен не к `prod`, а к кандидату.
            - Выбрали для выкатки r5 с лучшим F1, хотя он не прошёл smoke-тест и задержку.

            **Что вынести:** метрики сравнимы, только если посчитаны на одной зафиксированной выборке; гейт должен
            смотреть и на сегменты, и на эксплуатационные ограничения, а не только на одно среднее число.
            → шпаргалка «CI/CD для ML».
        """,
    },
    # ------------------------------------------------------------ feature_stores
    {
        "name": "feature_stores",
        "title": "Point-in-time correct join без утечки из будущего",
        "data": data_feature_stores,
        "task": """
            Соберите обучающую выборку так, как это делает `get_historical_features` в feature store.

            - `input.csv` — метки: `label_id, user_id, ts, churned` (момент, на который известна метка);
            - `orders_features.csv` — история значений признака `orders_30d`: `user_id, ts, orders_30d`;
            - `tickets_features.csv` — история признака `tickets_90d`: `user_id, ts, tickets_90d`.

            Правила (одинаковые для обучения и для онлайн-сервиса):

            1. Для каждой метки берётся **последнее значение признака того же пользователя с `ts` строго раньше**
               момента метки. Значение, записанное в тот же день, считается ещё недоступным (пакетный расчёт
               завершается в конце дня).
            2. У `orders_30d` есть **TTL = 30 дней**: если найденное значение старше 30 дней
               (разница дат > 30), оно считается устаревшим и не используется. У `tickets_90d` TTL нет.
            3. Если значения нет (или оно устарело) — запишите `-1`.
            4. `orders_age_days` — возраст использованного значения `orders_30d` в днях (`-1`, если значения нет).
            5. `leak` = 1, если «наивный» join — подставить пользователю **самое последнее** известное значение
               `orders_30d` из всей истории — дал бы другое число, чем point-in-time join (в том числе когда PIT даёт `-1`).

            ## Подсказки

            - Посмотрите на пример с `pd.merge_asof` в лекции: обе таблицы нужно отсортировать по `ts`,
              а соединять — `on='ts', by='user_id', direction='backward'`.
            - «Строго раньше» — параметр `allow_exact_matches=False`; TTL — `tolerance=pd.Timedelta('30D')`.
              Чтобы посчитать возраст значения, сохраните его дату отдельным столбцом до merge.
            - `merge_asof` меняет порядок строк — верните его сортировкой по `label_id`. Для проверки напишите
              медленный вариант циклом: для каждой метки отфильтруйте историю пользователя `ts < label_ts` и возьмите последнюю строку.

            ## Ожидаемый формат выходного файла

            CSV с колонками `label_id, user_id, orders_30d, orders_age_days, tickets_90d, leak` —
            16 строк по возрастанию `label_id`; все значения целые.
        """,
        "solution": """
            TTL = pd.Timedelta('30D')


            def pit_join(labels, feats, col, tolerance=None):
                feats = feats.rename(columns={'ts': 'feature_ts'}).assign(ts=lambda d: d['feature_ts'])
                return pd.merge_asof(labels.sort_values('ts'), feats.sort_values('ts'), on='ts', by='user_id',
                                     direction='backward', allow_exact_matches=False, tolerance=tolerance)


            def slow_value(hist, user, ts, col, ttl=None):
                h = hist[(hist['user_id'] == user) & (hist['ts'] < ts)].sort_values('ts')
                if h.empty or (ttl is not None and ts - h['ts'].iloc[-1] > ttl):
                    return -1
                return int(h[col].iloc[-1])


            def main():
                labels = pd.read_csv('input.csv', parse_dates=['ts'])
                orders = pd.read_csv('orders_features.csv', parse_dates=['ts'])
                tickets = pd.read_csv('tickets_features.csv', parse_dates=['ts'])

                o = pit_join(labels, orders, 'orders_30d', tolerance=TTL)
                o['orders_age_days'] = (o['ts'] - o['feature_ts']).dt.days
                t = pit_join(labels, tickets, 'tickets_90d')
                df = (o[['label_id', 'user_id', 'orders_30d', 'orders_age_days']]
                      .merge(t[['label_id', 'tickets_90d']], on='label_id')
                      .sort_values('label_id').reset_index(drop=True))
                df = df.fillna(-1).astype({'orders_30d': int, 'orders_age_days': int, 'tickets_90d': int})

                # «Наивный» join: последнее значение из всей истории — то, что лежит в онлайн-хранилище сейчас
                latest = orders.sort_values('ts').groupby('user_id')['orders_30d'].last()
                naive = labels.sort_values('label_id')['user_id'].map(latest).to_numpy()
                df['leak'] = (naive != df['orders_30d'].to_numpy()).astype(int)

                # Медленная проверка циклом
                for row, lab in zip(df.itertuples(), labels.sort_values('label_id').itertuples()):
                    assert row.orders_30d == slow_value(orders, lab.user_id, lab.ts, 'orders_30d', TTL)
                    assert row.tickets_90d == slow_value(tickets, lab.user_id, lab.ts, 'tickets_90d')

                print(f"строк с утечкой при наивном join: {df['leak'].sum()} из {len(df)}")
                return df
        """,
        "review": """
            **Идея.** Обучающая выборка должна содержать ровно то, что модель знала бы в момент предсказания.
            `merge_asof(direction='backward')` для каждой метки находит последнее значение признака до её момента;
            `allow_exact_matches=False` убирает значения, посчитанные в тот же день, а `tolerance` реализует TTL —
            так же, как онлайн-хранилище перестаёт отдавать устаревшие значения.

            **Типичные ошибки:**
            - Обычный `merge` по `user_id` и взятие последнего значения — признаки из будущего (столбец `leak` показывает,
              сколько таких меток): на валидации модель выглядит лучше, чем будет в бою.
            - `allow_exact_matches=True` (по умолчанию) — у меток, совпавших по дате с обновлением признака, значение «того же дня».
            - Не отсортировали таблицы по `ts` — `merge_asof` падает с ошибкой; не вернули порядок по `label_id`.
            - Забыли TTL — пользователь 5 получает значение двухмесячной давности, которого сервис уже не увидит.

            **Что вынести:** point-in-time join — главная функция офлайн-части feature store; правила доступности
            (задержка расчёта, TTL) должны быть одинаковыми для обучения и онлайна, иначе получится
            training-serving skew. → шпаргалка «Feature Stores».
        """,
    },
    # ------------------------------------------------------------ mlops_best_practices
    {
        "name": "mlops_best_practices",
        "title": "Трекинг экспериментов: версия данных по хешу и выбор лучшего запуска",
        "data": data_mlops,
        "task": """
            В `runs.jsonl` — журнал экспериментов (по одному JSON-объекту на строку, как в лекции): `run_id`, `status`,
            `params`, `metrics` (`cv_auc`, `cv_auc_std`, `test_auc`), `data_version`, `git_commit`.
            В `input.csv` — **текущая** версия обучающих данных. Часть запусков сделана на старой версии.

            1. Вычислите версию текущих данных так же, как её записывал трекер: прочитайте `input.csv` как текст
               (`Path('input.csv').read_text(encoding='utf-8')` — переводы строк при этом приводятся к `\\n`),
               возьмите MD5 от этого текста в UTF-8 и первые 8 шестнадцатеричных символов.
            2. `data_ok` = 1, если `data_version` запуска совпадает с текущей версией.
            3. `eligible` = 1, если `data_ok = 1` и `status == 'FINISHED'`.
            4. Среди допустимых запусков отранжируйте по `cv_auc` по убыванию; при равенстве выше тот,
               у кого меньше `cv_auc_std`, затем — меньший `run_id`. `rank` — место (1, 2, …), у недопустимых 0.
            5. `best` = 1 у запуска с `rank = 1`. `test_auc` для выбора **не используется**: отложенная выборка
               нужна для финальной оценки, а не для отбора.

            ## Подсказки

            - Журнал читается так: `[json.loads(line) for line in Path('runs.jsonl').read_text(encoding='utf-8').splitlines()]`,
              а `pd.json_normalize(runs)` развернёт вложенные `metrics` в столбцы `metrics.cv_auc` и т.д.
            - Хеш: `hashlib.md5(text.encode('utf-8')).hexdigest()[:8]`. Хеш от байтов файла (`read_bytes`) может
              отличаться из-за переводов строк Windows — поэтому трекер хеширует нормализованный текст.
            - Ранжирование с несколькими ключами: `sort_values(['cv_auc', 'cv_auc_std', 'run_id'], ascending=[False, True, True])`
              по допустимым запускам, затем `rank = 1..k` по полученному порядку.

            ## Ожидаемый формат выходного файла

            CSV с колонками `run_id, data_ok, eligible, rank, best` — по строке на каждый запуск в порядке журнала;
            все значения, кроме `run_id`, целые.
        """,
        "solution": """
            import hashlib
            import json
            from pathlib import Path


            def data_version(path):
                text = Path(path).read_text(encoding='utf-8')          # \\r\\n → \\n
                return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]


            def main():
                current = data_version('input.csv')
                runs = [json.loads(line) for line in Path('runs.jsonl').read_text(encoding='utf-8').splitlines() if line]
                df = pd.json_normalize(runs)
                print('текущая версия данных:', current)

                df['data_ok'] = (df['data_version'] == current).astype(int)
                df['eligible'] = ((df['data_ok'] == 1) & (df['status'] == 'FINISHED')).astype(int)

                ok = df[df['eligible'] == 1].sort_values(
                    ['metrics.cv_auc', 'metrics.cv_auc_std', 'run_id'], ascending=[False, True, True])
                df['rank'] = 0
                df.loc[ok.index, 'rank'] = np.arange(1, len(ok) + 1)
                df['best'] = (df['rank'] == 1).astype(int)

                best = df[df['best'] == 1].iloc[0]
                print('лучший запуск:', best['run_id'], best.filter(like='params.').dropna().to_dict(),
                      'cv_auc =', best['metrics.cv_auc'])
                trap = df.loc[df['metrics.cv_auc'].idxmax(), 'run_id']
                print('максимальный cv_auc во всём журнале у', trap, '— но он не допустим')
                return df[['run_id', 'data_ok', 'eligible', 'rank', 'best']]
        """,
        "review": """
            **Идея.** Журнал экспериментов полезен, только если по нему можно понять, *на чём* обучен запуск. Хеш
            содержимого данных (как у DVC) однозначно связывает запуск с версией датасета: метрики запусков на разных
            версиях данных сравнивать нельзя. Выбор лучшего — по метрике кросс-валидации среди успешных запусков на
            актуальных данных, с явными правилами разрешения ничьих.

            **Типичные ошибки:**
            - Выбрали run_04 с максимальным `cv_auc` — он упал (`FAILED`), метрика неполная.
            - Выбрали run_02 — его метрика выше, но получена на старой версии данных.
            - Выбрали run_09 по `test_auc` — это подгонка под отложенную выборку, оценка становится оптимистичной.
            - Не учли ничью `cv_auc` у run_06 и run_07 — побеждает более стабильный run_07 (меньше `cv_auc_std`).
            - Хешировали байты файла — на Windows `\\r\\n` даёт другой хеш, и ни один запуск не проходит `data_ok`.

            **Что вынести:** запуск = параметры + метрики + версия кода + версия данных; правило выбора модели
            фиксируется заранее и не использует тестовую выборку. → шпаргалка «MLOps Best Practices».
        """,
    },
    # ------------------------------------------------------------ model_monitoring_ab_testing
    {
        "name": "model_monitoring_ab_testing",
        "title": "Мониторинг: PSI и KS по признакам и z-тест A/B с доверительным интервалом",
        "data": data_monitoring,
        "task": """
            **Часть 1. Дрейф.** В `input.csv` строки с `sample = reference` (обучающее окно) и `sample = current`
            (последняя неделя), признаки `income`, `session_min`, `basket`. Для каждого признака:

            1. Границы 10 бинов — квантили референса: `edges = np.quantile(ref, np.linspace(0, 1, 11))`,
               затем крайние границы заменить на −∞ и +∞.
            2. Доли eᵢ (референс) и aᵢ (текущее окно) через `np.histogram(x, edges)[0] / len(x)`; к каждой доле
               прибавить ε = 1e-6.
            3. PSI = Σ (aᵢ − eᵢ) · ln(aᵢ / eᵢ).
            4. Статистика KS — `scipy.stats.ks_2samp(ref, cur).statistic`.

            **Часть 2. A/B-тест.** В `ab.csv` — пользователи (`group` = `A` старая модель, `B` новая) и `converted`.

            5. Конверсии p_A и p_B, разница d = p_B − p_A.
            6. z-тест для двух долей с **объединённой** долей p = (x_A + x_B)/(n_A + n_B):
               z = d / √(p(1 − p)(1/n_A + 1/n_B)), двусторонний p-value = 2·(1 − Φ(|z|)).
            7. 95%-доверительный интервал для d — с **раздельной** (unpooled) ошибкой:
               SE = √(p_A(1 − p_A)/n_A + p_B(1 − p_B)/n_B), d ± z₀.₉₇₅·SE, где z₀.₉₇₅ = `stats.norm.ppf(0.975)`.

            ## Подсказки

            - Границы бинов берутся **только по референсу** — тогда в каждом его бине ровно ~10% объектов,
              а сдвиг текущего окна виден как перекос долей.
            - Пример функции `psi` есть в лекции; ε нужен, чтобы ln не уходил в бесконечность на пустом бине.
            - Для A/B: `g = ab.groupby('group')['converted'].agg(['sum', 'count'])`. Проверьте себя: если 0 не входит в
              интервал, p-value должно быть < 0.05 (почти всегда — у z-теста и интервала разные SE).

            ## Ожидаемый формат выходного файла

            CSV с колонками `metric, value` и строками в таком порядке:
            `psi_income, psi_session_min, psi_basket, ks_income, ks_session_min, ks_basket, conv_a, conv_b, diff, z, p_value, ci_low, ci_high`;
            `value` округлить до 4 знаков.
        """,
        "solution": """
            from scipy import stats

            FEATURES = ['income', 'session_min', 'basket']


            def psi(ref, cur, bins=10, eps=1e-6):
                edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
                edges[0], edges[-1] = -np.inf, np.inf
                e = np.histogram(ref, edges)[0] / len(ref) + eps
                a = np.histogram(cur, edges)[0] / len(cur) + eps
                return float(np.sum((a - e) * np.log(a / e)))


            def main():
                df = pd.read_csv('input.csv')
                ref, cur = df[df['sample'] == 'reference'], df[df['sample'] == 'current']
                rows = []
                for f in FEATURES:
                    rows.append(('psi_' + f, psi(ref[f].to_numpy(), cur[f].to_numpy())))
                for f in FEATURES:
                    rows.append(('ks_' + f, stats.ks_2samp(ref[f], cur[f]).statistic))
                for name, value in rows[:3]:
                    level = 'стабильно' if value < 0.1 else 'умеренный сдвиг' if value < 0.25 else 'значимый сдвиг'
                    print(f"{name}: {value:.3f} — {level}")

                ab = pd.read_csv('ab.csv')
                g = ab.groupby('group')['converted'].agg(['sum', 'count'])
                (x_a, n_a), (x_b, n_b) = g.loc['A'], g.loc['B']
                p_a, p_b = x_a / n_a, x_b / n_b
                d = p_b - p_a
                p = (x_a + x_b) / (n_a + n_b)
                z = d / np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))          # SE при H0: объединённая доля
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)  # SE для интервала: раздельная
                q = stats.norm.ppf(0.975)
                rows += [('conv_a', p_a), ('conv_b', p_b), ('diff', d), ('z', z), ('p_value', p_value),
                         ('ci_low', d - q * se), ('ci_high', d + q * se)]
                print(f"A/B: {p_a:.2%} → {p_b:.2%}, z = {z:.2f}, p = {p_value:.4f}, "
                      f"95% ДИ [{d - q * se:+.4f}; {d + q * se:+.4f}]")

                result = pd.DataFrame(rows, columns=['metric', 'value'])
                result['value'] = result['value'].astype(float).round(4)
                return result
        """,
        "review": """
            **Идея.** Мониторинг входа сравнивает распределение признака в текущем окне с референсом. PSI суммирует
            перекос долей по бинам-децилям референса (< 0.1 — стабильно, 0.1–0.25 — умеренно, > 0.25 — значимо),
            KS — максимальное расстояние между эмпирическими функциями распределения. В A/B-тесте z-статистика
            считается при нулевой гипотезе (объединённая доля), а интервал описывает величину эффекта
            (раздельные дисперсии групп).

            **Типичные ошибки:**
            - Бины по объединённой выборке или по текущему окну — PSI другой и плохо сравним между неделями.
            - Без замены крайних границ на ±∞ значения текущего окна за пределами референса выпадают из гистограммы.
            - Перепутали направление разности (p_A − p_B) — у `diff`, `z` и интервала другой знак.
            - Интервал с объединённой SE или с 1.96 вместо точного квантиля — расхождение в 4-м знаке.
            - Односторонний p-value (без множителя 2).

            **Что вынести:** сдвиг `session_min` заметен по обеим метрикам, а у `basket` небольшой KS при большом n
            может быть «значим» статистически, но не практически — смотрите на величину, а не только на p-value.
            Длительность A/B-теста фиксируется заранее, без подглядывания. → шпаргалка «Model Monitoring и A/B Testing».
        """,
    },
    # ------------------------------------------------------------ model_serialization
    {
        "name": "model_serialization",
        "title": "Сериализация: pipeline, joblib/pickle, SHA-256 и проверка метаданных",
        "data": data_serialization,
        "task": """
            Дан файл `input.csv` (признаки `age, income, tenure, n_products`, цель `target`, `split`),
            реестр сохранённых моделей `registry.csv` и описание окружения сервиса `service_env.json`.

            1. Обучите на train `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`, передав признаки
               как DataFrame (в указанном порядке столбцов).
            2. Сохраните пайплайн двумя способами — `joblib.dump` и `pickle.dump` — во временную папку, загрузите обратно
               и сравните `predict_proba` на test с исходной моделью: `max_diff_joblib`, `max_diff_pickle` — максимум
               |разности| вероятностей класса 1.
            3. Метаданные: посчитайте SHA-256 файла joblib до загрузки; `sha_match` = 1, если хеш файла при повторном
               чтении совпал с записанным. Затем допишите в **копию** файла один байт `b'\\x00'`:
               `sha_tampered_match` = 1/0 — совпал ли хеш испорченной копии с записанным.
            4. `features_match` = 1, если `feature_names_in_` загруженной модели совпадает со списком `features`
               из `service_env.json` (с учётом порядка).
            5. `n_changed_without_scaler` — сколько предсказаний (`predict`) на test изменится, если по ошибке сохранить
               и применить только последний шаг пайплайна (LogisticRegression) к **немасштабированным** признакам
               (`pipe[-1].predict(X_test.to_numpy())`).
            6. Реестр: модель `compat_<model_id>` = 1, если она совместима с сервисом: совпадают **major.minor** версии
               scikit-learn, **major** версии NumPy и список признаков (с порядком, разделитель `|`). Версии сравнивайте как
               числа, а не как строки.
            7. Также выведите `test_accuracy` и `proba_sum` — сумму вероятностей класса 1 по test (контрольное число).

            ## Подсказки

            - Временная папка: `with tempfile.TemporaryDirectory() as d: path = Path(d) / 'model.joblib'`.
              Хеш файла: `hashlib.sha256(path.read_bytes()).hexdigest()`.
            - Почему сохраняют весь пайплайн? Последний шаг обучен на стандартизованных признаках; доход ~60 000
              без масштабирования «перевешивает» всё остальное.
            - Версию удобно разбирать так: `tuple(int(p) for p in v.split('.')[:2])` — тогда `1.19` ≠ `1.9`,
              а строковое сравнение `'1.19' > '1.9'` дало бы `False`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `check, value` и строками в таком порядке:
            `test_accuracy, proba_sum, max_diff_joblib, max_diff_pickle, sha_match, sha_tampered_match, features_match,
            n_changed_without_scaler, compat_m1, compat_m2, compat_m3, compat_m4, compat_m5, compat_m6`;
            `value` — число, округлённое до 4 знаков.
        """,
        "solution": """
            import hashlib
            import json
            import pickle
            import shutil
            import tempfile
            from pathlib import Path

            import joblib
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['age', 'income', 'tenure', 'n_products']


            def version(v, parts):
                return tuple(int(p) for p in str(v).split('.')[:parts])


            def main():
                df = pd.read_csv('input.csv')
                env = json.load(open('service_env.json', encoding='utf-8'))
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                X_tr, X_te = train[FEATURES], test[FEATURES]

                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, train['target'])
                proba = pipe.predict_proba(X_te)[:, 1]
                rows = [('test_accuracy', pipe.score(X_te, test['target'])), ('proba_sum', proba.sum())]

                with tempfile.TemporaryDirectory() as d:
                    jl, pk = Path(d) / 'model.joblib', Path(d) / 'model.pkl'
                    joblib.dump(pipe, jl)
                    with open(pk, 'wb') as f:
                        pickle.dump(pipe, f)
                    meta = {'sha256': hashlib.sha256(jl.read_bytes()).hexdigest(), 'features': FEATURES}

                    # --- «в сервисе» ---
                    sha_ok = hashlib.sha256(jl.read_bytes()).hexdigest() == meta['sha256']
                    bad = Path(d) / 'tampered.joblib'
                    shutil.copy(jl, bad)
                    with open(bad, 'ab') as f:
                        f.write(b'\\x00')
                    sha_bad = hashlib.sha256(bad.read_bytes()).hexdigest() == meta['sha256']

                    loaded_jl = joblib.load(jl)
                    with open(pk, 'rb') as f:
                        loaded_pk = pickle.load(f)

                diff_jl = np.abs(loaded_jl.predict_proba(X_te)[:, 1] - proba).max()
                diff_pk = np.abs(loaded_pk.predict_proba(X_te)[:, 1] - proba).max()
                features_ok = list(loaded_jl.feature_names_in_) == env['features']
                # Ошибка «сохранили только модель»: последний шаг на сырых признаках
                changed = int((pipe[-1].predict(X_te.to_numpy()) != pipe.predict(X_te)).sum())
                rows += [('max_diff_joblib', diff_jl), ('max_diff_pickle', diff_pk), ('sha_match', sha_ok),
                         ('sha_tampered_match', sha_bad), ('features_match', features_ok),
                         ('n_changed_without_scaler', changed)]

                reg = pd.read_csv('registry.csv')
                for r in reg.itertuples():
                    ok = (version(r.sklearn_version, 2) == version(env['sklearn_version'], 2)
                          and version(r.numpy_version, 1) == version(env['numpy_version'], 1)
                          and r.features.split('|') == env['features'])
                    rows.append((f'compat_{r.model_id}', ok))

                result = pd.DataFrame(rows, columns=['check', 'value'])
                result['value'] = result['value'].astype(float).round(4)
                print(result.to_string(index=False))
                return result
        """,
        "review": """
            **Идея.** Сохраняется не «модель», а весь объект предобработки и модели (`Pipeline`) — тогда в сервисе
            невозможно забыть масштабирование. Загрузка pickle/joblib в том же окружении даёт бит-в-бит те же
            предсказания; рядом с файлом хранят метаданные: SHA-256 (ловит повреждение и подмену даже на один байт),
            список признаков и версии библиотек, которые сверяются при старте сервиса.

            **Типичные ошибки:**
            - Сохранили только `LogisticRegression` — на сырых признаках меняется заметная доля предсказаний.
            - Сравнивали версии как строки (`'1.19.0'.startswith('1.1')`, `'1.19' < '1.9'`) — m6 легко ошибочно счесть совместимым.
            - Проверили только множество признаков, а не их порядок — m4 с переставленными столбцами «проходит»,
              хотя тихо даст неверные прогнозы.
            - Требовали точного совпадения версии патча (`1.9.0` ≠ `1.9.1`) — m1 отклонён, хотя по правилу задания совместим.
            - Загрузили pickle из непроверенного источника без сверки хеша — `pickle.load` может выполнить произвольный код.

            **Что вынести:** Pipeline + метаданные + контрольная сумма; после обновления библиотек модель переобучают
            или сверяют предсказания на контрольной выборке. → шпаргалка «Сериализация моделей ML».
        """,
    },
    # ------------------------------------------------------------ production_ml_best_practices
    {
        "name": "production_ml_best_practices",
        "title": "Задержка инференса: перцентили, ошибки и проверка SLA",
        "data": data_production,
        "task": """
            В `input.csv` — лог запросов к сервису инференса за сутки, когда две версии модели (`v1` и `v2`)
            обслуживали трафик в canary-режиме: `request_id, model_version, status` (`ok`, `timeout`, `error`) и `latency_ms`.
            При таймауте сервис ждал 300 мс и вернул запасной ответ — задержка таких запросов записана как 300.

            Для каждой версии:

            1. `n` — число запросов; `error_rate` — доля запросов со статусом не `ok` (таймауты тоже ошибки).
            2. `p50`, `p95`, `p99` задержки по **всем** запросам версии (включая таймауты и ошибки) —
               `np.percentile` с линейной интерполяцией (по умолчанию).
            3. SLA: p95 ≤ 150 мс, p99 ≤ 250 мс и `error_rate` ≤ 1%. `sla_ok` = 1, если выполнены все три условия.
            4. `budget_left` — остаток бюджета ошибок: 0.01·n − (число ошибок). Отрицательное значение — бюджет исчерпан.
            5. `mean_ms` — средняя задержка (для сравнения: среднее скрывает «хвост»).

            ## Подсказки

            - `df.groupby('model_version')['latency_ms'].quantile(0.95)` даёт то же, что `np.percentile(..., 95)`.
            - Не отбрасывайте таймауты при расчёте перцентилей: пользователь ждал эти 300 мс, и именно они формируют p99.
            - Сравните версии: у `v2` медиана ниже, но хвост тяжелее. Какую выкатили бы, глядя только на `mean_ms` или `p50`?

            ## Ожидаемый формат выходного файла

            CSV с колонками `model_version, n, error_rate, mean_ms, p50, p95, p99, sla_ok, budget_left` —
            2 строки в порядке `v1, v2`; дробные значения округлить до 4 знаков.
        """,
        "solution": """
            SLA_P95, SLA_P99, SLA_ERR = 150, 250, 0.01


            def main():
                df = pd.read_csv('input.csv')
                rows = []
                for ver, g in df.groupby('model_version'):
                    lat = g['latency_ms'].to_numpy()
                    n = len(g)
                    n_err = int((g['status'] != 'ok').sum())
                    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
                    error_rate = n_err / n
                    sla_ok = int(p95 <= SLA_P95 and p99 <= SLA_P99 and error_rate <= SLA_ERR)
                    rows.append({'model_version': ver, 'n': n, 'error_rate': error_rate, 'mean_ms': lat.mean(),
                                 'p50': p50, 'p95': p95, 'p99': p99, 'sla_ok': sla_ok,
                                 'budget_left': SLA_ERR * n - n_err})
                    # сверка с pandas
                    assert np.isclose(g['latency_ms'].quantile(0.99), p99)

                result = pd.DataFrame(rows)
                print(result.round(1).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** Пользовательский опыт определяется хвостом распределения задержки, поэтому SLA формулируют через
            перцентили (p95, p99), а не через среднее. Таймауты — часть распределения: исключив их, вы «улучшите»
            p99 ровно за счёт самых недовольных пользователей. Бюджет ошибок переводит SLA в понятное число:
            сколько ещё сбоев можно допустить.

            **Типичные ошибки:**
            - Считали перцентили только по `ok` — p99 у `v2` падает с ~258 до ~169 мс, и нарушение по задержке «пропадает».
            - Не считали таймауты ошибками — `error_rate` занижен.
            - Перцентиль с `method='nearest'` или по отсортированному массиву «на глаз» — небольшие расхождения.
            - Выбрали `v2` по медиане или среднему: она быстрее «в типичном случае», но нарушает p99 и бюджет ошибок.

            **Что вынести:** новую версию в продакшене сравнивают по p95/p99 и ошибкам на живом трафике (shadow/canary);
            таймаут с запасным ответом (бейзлайном) — обязательная часть надёжного сервиса.
            → шпаргалка «Production ML Best Practices».
        """,
    },
    # ------------------------------------------------------------ fairness_ml
    {
        "name": "fairness_ml",
        "title": "Метрики справедливости и выравнивание TPR порогом",
        "data": data_fairness,
        "task": """
            В `input.csv` — решения кредитной модели: `group` (защищённый признак A ∈ {0, 1}), `y_true`
            (1 — клиент вернул бы кредит), `score` — вероятность от модели. Решение: Ŷ = 1, если `score ≥ порог`.

            **Сценарий `base`:** общий порог 0.5 для обеих групп.
            **Сценарий `adjusted`:** для группы 1 порог остаётся 0.5, а порог группы 0 выбирается из сетки
            0.20, 0.21, …, 0.70 так, чтобы TPR группы 0 был **как можно ближе** к TPR группы 1 из сценария `base`
            (минимум |TPR₀(t) − TPR₁|; при равенстве — больший порог). Это пост-обработка в духе `ThresholdOptimizer`.

            Для каждого сценария посчитайте по группам g ∈ {0, 1}:

            - `sr_g` — доля одобрений P(Ŷ = 1 | A = g);
            - `tpr_g` = P(Ŷ = 1 | Y = 1, A = g), `fpr_g` = P(Ŷ = 1 | Y = 0, A = g);

            и сводные показатели:

            - `dp_diff` = sr₁ − sr₀ (demographic parity difference), `di_ratio` = sr₀ / sr₁ (disparate impact, правило 4/5);
            - `eo_gap` = max(|tpr₁ − tpr₀|, |fpr₁ − fpr₀|) (разрыв equalized odds);
            - `accuracy` — доля верных решений по всем клиентам.

            ## Подсказки

            - Удобна функция `rates(pred, y, mask)`, возвращающая `pred[mask].mean()`, `pred[mask & (y == 1)].mean()`,
              `pred[mask & (y == 0)].mean()`.
            - Сетку порогов стройте через `np.round(np.arange(20, 71) / 100, 2)` — так `0.2 + 0.01·k` не даст
              «0.31000000000000005», и сравнение `score ≥ t` будет точным.
            - Выбор порога с правилом «при равенстве больший»: переберите сетку от большего к меньшему и обновляйте
              лучший, только если разрыв **строго** меньше.

            ## Ожидаемый формат выходного файла

            CSV с колонками `scenario, thr_0, thr_1, sr_0, sr_1, tpr_0, tpr_1, fpr_0, fpr_1, dp_diff, di_ratio, eo_gap, accuracy` —
            2 строки в порядке `base, adjusted`; значения округлить до 4 знаков.
        """,
        "solution": """
            def rates(pred, y, mask):
                return pred[mask].mean(), pred[mask & (y == 1)].mean(), pred[mask & (y == 0)].mean()


            def evaluate(df, thr0, thr1, name):
                y, g, s = df['y_true'].to_numpy(), df['group'].to_numpy(), df['score'].to_numpy()
                thr = np.where(g == 0, thr0, thr1)
                pred = (s >= thr).astype(int)
                sr0, tpr0, fpr0 = rates(pred, y, g == 0)
                sr1, tpr1, fpr1 = rates(pred, y, g == 1)
                return {'scenario': name, 'thr_0': thr0, 'thr_1': thr1, 'sr_0': sr0, 'sr_1': sr1,
                        'tpr_0': tpr0, 'tpr_1': tpr1, 'fpr_0': fpr0, 'fpr_1': fpr1,
                        'dp_diff': sr1 - sr0, 'di_ratio': sr0 / sr1,
                        'eo_gap': max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)), 'accuracy': (pred == y).mean()}


            def main():
                df = pd.read_csv('input.csv')
                base = evaluate(df, 0.5, 0.5, 'base')

                # Порог группы 0, выравнивающий TPR с группой 1 (при равенстве — больший порог)
                y0 = df.loc[df['group'] == 0, 'y_true'].to_numpy()
                s0 = df.loc[df['group'] == 0, 'score'].to_numpy()
                best_t, best_gap = None, np.inf
                for t in np.round(np.arange(70, 19, -1) / 100, 2):          # от большего к меньшему
                    gap = abs((s0[y0 == 1] >= t).mean() - base['tpr_1'])
                    if gap < best_gap - 1e-12:
                        best_t, best_gap = t, gap
                adjusted = evaluate(df, float(best_t), 0.5, 'adjusted')

                result = pd.DataFrame([base, adjusted])
                print(result[['scenario', 'thr_0', 'di_ratio', 'eo_gap', 'accuracy']].round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** Групповые метрики справедливости сравнивают долю одобрений (demographic parity) и ошибки
            по группам (equalized odds: TPR и FPR). Здесь скоры группы 1 систематически завышены, поэтому при общем
            пороге группа 0 одобряется реже даже среди тех, кто вернул бы кредит. Пост-обработка — отдельный порог
            для группы 0 — выравнивает TPR (equal opportunity), но обычно меняет FPR и общую точность.

            **Типичные ошибки:**
            - TPR как доля одобренных среди всей группы (это sr), а не среди `y_true = 1`.
            - `di_ratio` = sr₁/sr₀ вместо sr₀/sr₁ — значение больше 1, правило 4/5 не читается.
            - Сетка через `np.arange(0.30, 0.71, 0.01)` без округления — пороги вида 0.35000000000000003 и сдвиг на граничных скорах.
            - Строгое `score > t` вместо `≥` — на скорах, совпадающих с порогом, решения другие.
            - Нет правила ничьих — при одинаковом разрыве выбран меньший порог.

            **Что вынести:** выравнивание одной метрики (TPR) не выравнивает другие (DP, FPR) — это прямое следствие
            теоремы о невозможности; выбор метрики — решение не только техническое, и цену (accuracy) нужно
            показывать открыто. → шпаргалка «Fairness в ML».
        """,
    },
    # ------------------------------------------------------------ explainable_ai_xai
    {
        "name": "explainable_ai_xai",
        "title": "Permutation importance вручную и сравнение с MDI",
        "data": data_xai,
        "task": """
            Дан файл `input.csv`: признаки `x1…x4`, цель `y` и `split`. Признаки `x1` и `x3` — два зашумлённых измерения одной и той же скрытой величины (сильно коррелируют), `x4` — шум.

            1. Обучите на train `RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)`.
            2. `mdi` — встроенная важность `feature_importances_` (уменьшение неоднородности, по train).
            3. **Permutation importance на test вручную:** базовое качество — R² модели на test. Для каждого признака
               j в порядке `x1, x2, x3, x4` и для каждого повтора r = 1…5 перемешайте **только столбец j** тестовой
               матрицы перестановкой `perm = rng.permutation(n_test)` и посчитайте падение R²: base − R²(перемешанные).
               Генератор один на весь расчёт: `rng = np.random.default_rng(42)`, создаётся один раз до циклов,
               порядок вызовов — внешний цикл по признакам, внутренний по повторам.
            4. `perm_mean`, `perm_std` — среднее и стандартное отклонение (`ddof=0`) пяти падений;
               `rank_perm` — место признака по `perm_mean` (1 — самый важный).

            ## Подсказки

            - Перемешивание разрывает связь признака с целью, сохраняя его распределение. Важен признак, без которого
              модель ошибается сильнее.
            - Копируйте матрицу перед перемешиванием: `Xp = X_test.copy(); Xp[:, j] = X_test[perm, j]`.
            - Сравните ранги `mdi` и `perm_mean` для `x1` и `x3`: коррелированные признаки делят важность, и методы
              делят её по-разному. `sklearn.inspection.permutation_importance` даст близкие, но не те же числа
              (у неё своя схема случайных перестановок).

            ## Ожидаемый формат выходного файла

            CSV с колонками `feature, mdi, perm_mean, perm_std, rank_perm` — 4 строки в порядке `x1, x2, x3, x4`;
            значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import r2_score

            FEATURES = ['x1', 'x2', 'x3', 'x4']
            N_REPEATS = 5


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                X_tr, y_tr = train[FEATURES].to_numpy(), train['y'].to_numpy()
                X_te, y_te = test[FEATURES].to_numpy(), test['y'].to_numpy()

                model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0).fit(X_tr, y_tr)
                base = r2_score(y_te, model.predict(X_te))
                print(f"R² на test: {base:.4f}")

                rng = np.random.default_rng(42)
                drops = np.zeros((len(FEATURES), N_REPEATS))
                for j in range(len(FEATURES)):
                    for r in range(N_REPEATS):
                        perm = rng.permutation(len(X_te))
                        Xp = X_te.copy()
                        Xp[:, j] = X_te[perm, j]                 # перемешан только признак j
                        drops[j, r] = base - r2_score(y_te, model.predict(Xp))

                perm_mean = drops.mean(axis=1)
                result = pd.DataFrame({'feature': FEATURES, 'mdi': model.feature_importances_,
                                       'perm_mean': perm_mean, 'perm_std': drops.std(axis=1)})
                result['rank_perm'] = result['perm_mean'].rank(ascending=False, method='first').astype(int)

                # Для сравнения — библиотечная реализация (другие перестановки, близкие значения)
                lib = permutation_importance(model, X_te, y_te, n_repeats=N_REPEATS, random_state=0)
                print('sklearn permutation_importance:', lib.importances_mean.round(3))
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** Permutation importance — модель-агностичный глобальный метод: насколько ухудшится качество на
            отложенной выборке, если «сломать» признак перемешиванием. MDI считается по обучающим данным внутри деревьев
            и отвечает на другой вопрос — как часто и с какой пользой признак используется в разбиениях.

            **Типичные ошибки:**
            - Перемешали всю строку матрицы (все признаки разом) или переобучали модель после перемешивания.
            - Создавали новый `default_rng(42)` для каждого признака — перестановки другие, числа не совпадут.
            - `std` с `ddof=1` (по умолчанию в pandas) — `perm_std` в √(5/4) раза больше.
            - Считали важность на train — на нём лес переобучен, и даже шумовой `x4` выглядит полезнее.

            **Что вынести:** у коррелированных `x1` и `x3` важность делится: перемешав `x1`, модель частично
            «спасается» через `x3`, поэтому оба выглядят менее важными, чем их общая информация. Объяснение описывает
            модель, а не причинность, и разные методы дают разные ранги. → шпаргалка «Explainable AI (XAI)».
        """,
    },
    # ------------------------------------------------------------ lime
    {
        "name": "lime",
        "title": "LIME вручную: взвешенный локальный линейный суррогат",
        "data": data_lime,
        "task": """
            «Чёрный ящик» задан формулой f(x) = σ(2·x₁² + x₂ − x₁·x₃ − 1), где σ(t) = 1 / (1 + e⁻ᵗ).
            В `input.csv` — три объекта для объяснения (`id, x1, x2, x3`), в `noise.csv` — 1000 строк стандартного
            нормального шума `e1, e2, e3` (общие для всех объектов, чтобы результат был воспроизводимым).
            Масштабы признаков (std на обучении): s = (1.0, 0.5, 2.0).

            Для каждого объекта x₀ выполните шаги LIME:

            1. Возмущения: zₖ = x₀ + eₖ ⊙ s (поэлементно); стандартизованные смещения — сами eₖ.
            2. Ответы чёрного ящика: pₖ = f(zₖ).
            3. Веса близости: πₖ = exp(−‖eₖ‖² / σ²), ширина ядра σ = 0.75·√3 (по умолчанию библиотеки lime).
            4. Взвешенная линейная регрессия p ≈ w₀ + w₁·e₁ + w₂·e₂ + w₃·e₃ **без регуляризации**, с весами π
               (взвешенный МНК; совпадает с `LinearRegression().fit(E, p, sample_weight=π)`).
            5. Качество суррогата — взвешенный R²: 1 − Σπ(p − p̂)² / Σπ(p − p̄_π)², где p̄_π — взвешенное среднее p.

            ## Подсказки

            - Коэффициент wⱼ — изменение вероятности при сдвиге признака j на одно стандартное отклонение вблизи x₀.
              Для гладкой f и узкого ядра он близок к ∂f/∂xⱼ · sⱼ в точке x₀. Сравните: при широком ядре (как здесь
              у объекта 1) расхождение заметно — суррогат усредняет наклон по всей окрестности.
            - Взвешенный МНК через обычный: умножьте строки [1, E] и p на √π и решите `np.linalg.lstsq`.
            - `LinearRegression(...).score(E, p, sample_weight=π)` возвращает ровно взвешенный R² из шага 5.
              Объект 3 лежит в сильно нелинейной области — посмотрите, насколько там ниже R².

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, f_x0, intercept, w1, w2, w3, r2` — 3 строки в порядке `id`;
            `f_x0` = f(x₀); значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LinearRegression

            SCALE = np.array([1.0, 0.5, 2.0])
            KERNEL_WIDTH = 0.75 * np.sqrt(3)


            def black_box(X):
                t = 2 * X[:, 0] ** 2 + X[:, 1] - X[:, 0] * X[:, 2] - 1
                return 1 / (1 + np.exp(-t))


            def main():
                points = pd.read_csv('input.csv')
                E = pd.read_csv('noise.csv')[['e1', 'e2', 'e3']].to_numpy()
                rows = []
                for pt in points.itertuples():
                    x0 = np.array([pt.x1, pt.x2, pt.x3])
                    Z = x0 + E * SCALE                                   # 1) возмущения
                    p = black_box(Z)                                     # 2) ответы чёрного ящика
                    pi = np.exp(-(E ** 2).sum(axis=1) / KERNEL_WIDTH ** 2)   # 3) веса близости

                    # 4) взвешенный МНК: строки умножаем на √π
                    A = np.column_stack([np.ones(len(E)), E])
                    sw = np.sqrt(pi)
                    coef, *_ = np.linalg.lstsq(A * sw[:, None], p * sw, rcond=None)
                    p_hat = A @ coef
                    p_bar = np.average(p, weights=pi)
                    r2 = 1 - np.sum(pi * (p - p_hat) ** 2) / np.sum(pi * (p - p_bar) ** 2)   # 5)

                    lr = LinearRegression().fit(E, p, sample_weight=pi)
                    assert np.allclose(lr.coef_, coef[1:]) and np.isclose(lr.intercept_, coef[0])
                    assert np.isclose(lr.score(E, p, sample_weight=pi), r2)

                    # Проверка смысла: градиент f в x0, умноженный на масштаб признаков
                    f0 = black_box(x0[None])[0]
                    grad = f0 * (1 - f0) * np.array([4 * x0[0] - x0[2], 1.0, -x0[0]]) * SCALE
                    print(f"id={pt.id}: LIME {coef[1:].round(3)}  ∇f·s {grad.round(3)}  R² {r2:.3f}")

                    rows.append({'id': pt.id, 'f_x0': f0, 'intercept': coef[0],
                                 'w1': coef[1], 'w2': coef[2], 'w3': coef[3], 'r2': r2})
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** LIME приближает сложную модель простой линейной *в окрестности* объекта: генерирует возмущения,
            спрашивает чёрный ящик, взвешивает точки ядром по близости и решает взвешенный МНК. Коэффициенты — локальные
            вклады признаков на единицу стандартного отклонения; взвешенный R² показывает, можно ли доверять объяснению.

            **Типичные ошибки:**
            - Регрессия на zₖ вместо стандартизованных смещений eₖ — коэффициенты в других единицах (делятся на sⱼ).
            - Расстояние в ядре по исходным признакам, а не по стандартизованным — признак `x3` с большим масштабом «съедает» веса.
            - Обычный (невзвешенный) МНК или `Ridge` по умолчанию — коэффициенты отличаются в 2–3 знаке.
            - Невзвешенный R² — завышает или занижает качество локального приближения.

            **Что вынести:** у объектов 1 и 2 знак вклада `x1` разный — глобально признак нелинейный, а LIME честно
            описывает только окрестность. Ширина ядра и схема возмущений меняют ответ, поэтому проверяйте R²
            и устойчивость. → шпаргалка «LIME: Интерпретация моделей».
        """,
    },
    # ------------------------------------------------------------ multi_armed_bandits
    {
        "name": "multi_armed_bandits",
        "title": "Многорукие бандиты: ε-greedy, UCB1 и Thompson на одном потоке наград",
        "data": data_bandits,
        "task": """
            Четыре баннера (руки 0–3) с истинными конверсиями μ = (0.10, 0.15, 0.22, 0.18) — агенту они неизвестны.
            В `input.csv` для каждого шага t = 1…3000 заранее записаны: награда каждой руки `r0…r3` (0/1) — если на шаге t
            выбрана рука a, агент получает `r{a}`; `u` и `rand_arm` — случайные числа для ε-greedy.
            Так все стратегии видят один и тот же «мир», и результат полностью воспроизводим.

            Стратегии (n(a) — число выборов руки, s(a) — сумма её наград, Q̂(a) = s(a)/n(a); везде при равенстве — меньший номер руки, как `np.argmax`):

            1. **ε-greedy, ε = 0.1.** На шагах 1–4 выбрать руки 0, 1, 2, 3 по очереди. Далее: если `u < 0.1` — рука `rand_arm`,
               иначе argmax Q̂(a).
            2. **UCB1.** Шаги 1–4 — руки 0, 1, 2, 3 по очереди. Далее argmax [Q̂(a) + √(2·ln t / n(a))], t — номер текущего шага.
            3. **Thompson Sampling.** `rng = np.random.default_rng(2026)` один раз до цикла; на **каждом** шаге (с первого)
               одна выборка `theta = rng.beta(1 + s, 1 + n - s)` для вектора всех рук, рука = argmax theta.

            Для каждой стратегии посчитайте `total_reward` (сумма полученных наград), `regret` = Σₜ (μ* − μ(aₜ))
            по **истинным** μ (μ* = 0.22), `best_share` — доля шагов, где выбрана лучшая рука 2, и `pulls_0…pulls_3`.

            ## Подсказки

            - Общий каркас: `run(choose)` с массивами `n = np.zeros(4)`, `s = np.zeros(4)`; на шаге t `a = choose(t, n, s)`,
              затем `s[a] += r[t, a]`, `n[a] += 1`, `regret += mu.max() - mu[a]`.
            - Regret считается по истинным μ, а не по полученным наградам — так он не зависит от «везения» на конкретных 0/1.
            - Для Thompson важен порядок вызовов генератора: ровно один вызов `rng.beta` с векторами на шаг, без предварительного
              раунда по всем рукам.

            ## Ожидаемый формат выходного файла

            CSV с колонками `strategy, total_reward, regret, best_share, pulls_0, pulls_1, pulls_2, pulls_3` —
            3 строки в порядке `eps_greedy, ucb1, thompson`; `regret` и `best_share` округлить до 4 знаков.
        """,
        "solution": """
            MU = np.array([0.10, 0.15, 0.22, 0.18])
            K, EPS = 4, 0.1


            def run(choose, df, rng=None):
                R = df[['r0', 'r1', 'r2', 'r3']].to_numpy()
                n, s = np.zeros(K), np.zeros(K)
                regret, total = 0.0, 0
                for i, row in enumerate(df.itertuples()):
                    t = i + 1
                    a = choose(t, n, s, row, rng)
                    s[a] += R[i, a]
                    n[a] += 1
                    total += R[i, a]
                    regret += MU.max() - MU[a]
                return total, regret, n


            def eps_greedy(t, n, s, row, rng):
                if t <= K:
                    return t - 1                                   # стартовый раунд
                if row.u < EPS:
                    return int(row.rand_arm)                       # исследуем
                return int(np.argmax(s / n))                       # используем


            def ucb1(t, n, s, row, rng):
                if t <= K:
                    return t - 1
                return int(np.argmax(s / n + np.sqrt(2 * np.log(t) / n)))


            def thompson(t, n, s, row, rng):
                return int(np.argmax(rng.beta(1 + s, 1 + n - s)))   # сэмпл из апостериорного Beta


            def main():
                df = pd.read_csv('input.csv')
                rows = []
                for name, f, rng in [('eps_greedy', eps_greedy, None), ('ucb1', ucb1, None),
                                     ('thompson', thompson, np.random.default_rng(2026))]:
                    total, regret, n = run(f, df, rng)
                    rows.append({'strategy': name, 'total_reward': int(total), 'regret': regret,
                                 'best_share': n[2] / len(df), **{f'pulls_{k}': int(n[k]) for k in range(K)}})
                result = pd.DataFrame(rows)
                print(result.round(3).to_string(index=False))
                print('случайный выбор дал бы regret ≈', round(len(df) * (MU.max() - MU.mean()), 1))
                return result.round(4)
        """,
        "review": """
            **Идея.** Бандит балансирует исследование и использование. ε-greedy исследует вслепую с постоянной частотой,
            поэтому его regret растёт линейно (~ε·T·средний разрыв). UCB1 добавляет бонус неопределённости √(2 ln t / n),
            который сам угасает у часто выбираемых рук; Thompson исследует, сэмплируя правдоподобные средние из Beta-апостериора.

            **Типичные ошибки:**
            - Regret по полученным наградам (T·μ* − total_reward) — число шумное и не совпадает с эталоном.
            - В UCB1 `ln(t − 1)` или `ln(Σn)` до увеличения счётчика вместо ln t текущего шага — выбор чуть другой.
            - Лишний стартовый раунд у Thompson или вызовы `rng.beta` по одной руке в цикле — другая
              последовательность случайных чисел.
            - ε-greedy: `u ≤ 0.1` вместо `u < 0.1`, или случайная рука через свой генератор вместо `rand_arm`.

            **Что вынести:** на одном и том же потоке наград видно, как стратегии распределяют показы; UCB1 с бонусом
            для наград в [0, 1] при маленьких конверсиях исследует долго — коэффициент стоит согласовывать с масштабом
            наград. → шпаргалка «Multi-Armed Bandits».
        """,
    },
    # ------------------------------------------------------------ adversarial_robustness
    {
        "name": "adversarial_robustness",
        "title": "FGSM на логистической регрессии вручную и сертифицированная устойчивость",
        "data": data_adversarial,
        "task": """
            В `input.csv` — рукописные цифры «1» (`label = 0`) и «7» (`label = 1`) из `load_digits`: 64 пикселя `p0…p63`
            в диапазоне [0, 1] и `split`.

            1. Обучите на train `LogisticRegression(max_iter=2000)` (C = 1 по умолчанию). w — `coef_[0]`, b — `intercept_[0]`.
            2. **FGSM вручную** на test: градиент log-loss по входу ∂L/∂x = (p − y)·w, где p = σ(w·x + b).
               x_adv = clip(x + ε·sign(∂L/∂x), 0, 1) для ε ∈ {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3}.
            3. Для каждого ε: `acc_fgsm` — accuracy на x_adv; `logloss` — средний log-loss на x_adv;
               `flip_rate` — доля test-объектов, у которых предсказание на x_adv отличается от предсказания на чистом x.
            4. `acc_noclip` — accuracy FGSM **без** `clip` (атакующему разрешён выход за [0, 1]).
            5. **Сертифицированная точность** для линейной модели: при любом возмущении ‖δ‖∞ ≤ ε логит меняется не больше
               чем на ε·‖w‖₁. Объект устойчив, если он классифицирован верно и |w·x + b| > ε·‖w‖₁.
               `acc_cert` — доля таких объектов.

            ## Подсказки

            - Градиент считается по **входу**, а не по весам: модель не меняется, меняется картинка. Для всех объектов сразу —
              `grad = (p - y)[:, None] * w`.
            - Без `clip` FGSM для линейной модели — **оптимальная** L∞-атака: логит каждого объекта сдвигается ровно на
              ε·‖w‖₁ в сторону ошибки. Поэтому `acc_noclip` должен совпасть с `acc_cert`.
            - `clip` ослабляет атаку (пиксель 0 нельзя сделать ещё темнее), поэтому `acc_fgsm` ≥ `acc_noclip`.
              log-loss удобно считать через `sklearn.metrics.log_loss(y, p_adv, labels=[0, 1])`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `eps, acc_fgsm, logloss, flip_rate, acc_noclip, acc_cert` — 7 строк по возрастанию ε;
            значения округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import log_loss

            EPSILONS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


            def main():
                df = pd.read_csv('input.csv')
                pix = [f'p{j}' for j in range(64)]
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
                X, y = test[pix].to_numpy(), test['label'].to_numpy()

                clf = LogisticRegression(max_iter=2000).fit(train[pix].to_numpy(), train['label'])
                w, b = clf.coef_[0], clf.intercept_[0]
                sigmoid = lambda t: 1 / (1 + np.exp(-t))

                p = sigmoid(X @ w + b)
                assert np.allclose(p, clf.predict_proba(X)[:, 1])
                clean_pred = (X @ w + b > 0).astype(int)
                grad = (p - y)[:, None] * w                      # ∂(log-loss)/∂x для каждого объекта
                margin = np.abs(X @ w + b)
                l1 = np.abs(w).sum()

                rows = []
                for eps in EPSILONS:
                    X_adv = np.clip(X + eps * np.sign(grad), 0, 1)
                    X_nc = X + eps * np.sign(grad)
                    p_adv = sigmoid(X_adv @ w + b)
                    pred_adv = (p_adv > 0.5).astype(int)
                    acc_nc = ((X_nc @ w + b > 0).astype(int) == y).mean()
                    cert = ((clean_pred == y) & (margin > eps * l1)).mean()
                    assert np.isclose(acc_nc, cert)              # FGSM без clip — оптимальная атака на линейную модель
                    rows.append({'eps': eps, 'acc_fgsm': (pred_adv == y).mean(),
                                 'logloss': log_loss(y, p_adv, labels=[0, 1]),
                                 'flip_rate': (pred_adv != clean_pred).mean(),
                                 'acc_noclip': acc_nc, 'acc_cert': cert})

                result = pd.DataFrame(rows)
                print(f"‖w‖₁ = {l1:.2f}; медианный «запас» |логит|/‖w‖₁ = {np.median(margin / l1):.3f}")
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** FGSM делает один шаг по знаку градиента потерь **по входу**. Для логистической регрессии градиент
            выписывается явно, (p − y)·w, его знак определяется только знаком веса и классом объекта: каждый пиксель сдвигается
            на ε в сторону, уменьшающую уверенность в верном классе. Сумма маленьких сдвигов по 64 пикселям меняет логит на ε·‖w‖₁ — это и есть
            «проклятие высокой размерности»: чем больше признаков, тем меньше нужен ε.

            **Типичные ошибки:**
            - Знак наоборот (x − ε·sign(∇)) — атака помогает модели, точность растёт.
            - Градиент по весам вместо входа или sign(w) без множителя (p − y) — для объектов класса 0 направление неверное.
            - Забыли `clip` — `acc_fgsm` совпадает с `acc_noclip` (картинки с пикселями вне [0, 1] физически невозможны).
            - `flip_rate` относительно истинных меток, а не чистых предсказаний — на ε = 0 должно быть 0.

            **Что вынести:** для линейных моделей устойчивость считается точно через запас |логит|/‖w‖₁; для нейросетей
            так нельзя, поэтому проверяют сильными атаками (PGD) и защищаются adversarial training.
            → шпаргалка «Adversarial Robustness».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям: данные, MLOps, production, надёжный и объяснимый ML:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=721):
        build(exercise, seed)


if __name__ == "__main__":
    main()
