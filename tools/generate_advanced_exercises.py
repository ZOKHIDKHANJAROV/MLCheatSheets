#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к Путям 3 и 4: production ML, интерпретируемость, NLP, временные ряды, RL.

Все ответы детерминированы и проверяются так же, как остальные упражнения:
    python tools/check_exercise.py exercises/<имя>

Запуск из корня репозитория:
    python tools/generate_advanced_exercises.py
"""

import numpy as np
import pandas as pd

from generate_basic_exercises import build

# ---------------------------------------------------------------- данные

def data_tuning(rng):
    n = 150
    X = rng.uniform(-2, 2, (n, 4)).round(3)
    y = np.sin(2 * X[:, 0]) * 3 + X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.8, n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["y"] = y.round(3)
    return df


def data_leakage(rng):
    n, p = 100, 500
    X = rng.normal(0, 1, (n, p)).round(3)                 # чистый шум: связи с целевой нет
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = rng.integers(0, 2, n)
    return df


def data_drift(rng):
    n = 1000
    ref = pd.DataFrame({"sample": "reference", "age": rng.normal(40, 10, n), "income": rng.lognormal(10, 0.5, n),
                        "sessions": rng.poisson(5, n)})
    cur = pd.DataFrame({"sample": "current", "age": rng.normal(40.5, 10, n), "income": rng.lognormal(10.15, 0.5, n),
                        "sessions": rng.poisson(8, n)})
    df = pd.concat([ref, cur], ignore_index=True)
    df["age"] = df["age"].round(1)
    df["income"] = df["income"].round(0)
    return df


def data_shap(rng):
    background = pd.DataFrame(rng.normal([1, 2, 0], [1, 1, 1], (200, 3)).round(3), columns=["x1", "x2", "x3"])
    explain = pd.DataFrame(rng.normal([1, 2, 0], [1.5, 1.5, 1.5], (6, 3)).round(2), columns=["x1", "x2", "x3"])
    explain.insert(0, "id", np.arange(1, 7))
    return {"input.csv": explain, "background.csv": background}


def data_tfidf(rng):
    docs = [
        "модель учится на данных и делает прогноз",
        "нейросеть учится на больших данных",
        "прогноз погоды на завтра",
        "градиентный бустинг строит деревья решений",
        "модель бустинг и модель нейросеть дают прогноз",
        "деревья решений легко интерпретировать",
    ]
    return pd.DataFrame({"doc": np.arange(1, len(docs) + 1), "text": docs})


def data_smoothing(rng):
    t = np.arange(24)
    y = 100 + 2.5 * t + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 24)
    return pd.DataFrame({"t": t, "y": y.round(2)})


def data_trajectories(rng):
    # Коридор из 5 клеток: 0 — старт, 4 — цель (+1, конец эпизода). Действия: 0 — влево, 1 — вправо.
    rows = []
    for ep in range(1, 31):
        s, step = 0, 0
        a = int(rng.random() < 0.6)
        while True:
            s2 = max(0, min(4, s + (1 if a == 1 else -1)))
            done = s2 == 4
            r = 1.0 if done else -0.1
            a2 = -1 if done else int(rng.random() < 0.6)
            rows.append({"episode": ep, "step": step, "s": s, "a": a, "r": r, "s_next": s2, "a_next": a2, "done": int(done)})
            if done or step >= 30:
                break
            s, a, step = s2, a2, step + 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "hyperparameter_tuning",
        "title": "Подбор глубины дерева кросс-валидацией",
        "data": data_tuning,
        "task": """
            Дан файл `input.csv`: признаки `x1..x4` и целевая переменная `y` (150 объектов).

            1. Для каждой глубины `max_depth` от 1 до 8 оцените `DecisionTreeRegressor(max_depth=..., random_state=0)`
               кросс-валидацией `KFold(n_splits=5, shuffle=True, random_state=42)` по метрике MSE
               (среднее MSE по 5 фолдам).
            2. Отметьте лучшую глубину — с минимальным средним MSE.

            ## Подсказки

            - `cross_val_score(..., scoring='neg_mean_squared_error')` возвращает MSE со знаком минус.
            - То же самое делает `GridSearchCV(param_grid={'max_depth': range(1, 9)}, cv=kf)` — сравните результаты.
            - Посмотрите на форму кривой: маленькая глубина — недообучение, большая — переобучение.
            - Лучший средний MSE на CV — оптимистичная оценка; честную даёт отдельный тест или вложенная CV.

            ## Ожидаемый формат выходного файла

            CSV с колонками `max_depth, cv_mse, is_best` (8 строк, глубина по возрастанию; `cv_mse` округлить до 4 знаков,
            `is_best` = 1 только у лучшей глубины, у остальных 0).
        """,
        "solution": """
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.tree import DecisionTreeRegressor


            def main():
                df = pd.read_csv('input.csv')
                X, y = df[['x1', 'x2', 'x3', 'x4']], df['y']
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                rows = []
                for depth in range(1, 9):
                    scores = cross_val_score(DecisionTreeRegressor(max_depth=depth, random_state=0), X, y,
                                             cv=kf, scoring='neg_mean_squared_error')
                    rows.append({'max_depth': depth, 'cv_mse': -scores.mean()})
                result = pd.DataFrame(rows)
                result['is_best'] = (result['cv_mse'] == result['cv_mse'].min()).astype(int)
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
    },
    {
        "name": "ml_pipelines",
        "title": "Утечка данных: отбор признаков вне пайплайна",
        "data": data_leakage,
        "task": """
            В файле `input.csv` 100 объектов, 500 признаков `f0..f499` и бинарная `target`.
            **Признаки — чистый случайный шум**: никакой связи с целевой переменной нет, честная accuracy ≈ 0.5.

            Оцените accuracy логистической регрессии кросс-валидацией
            `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` двумя способами:

            1. **leaky**: сначала `SelectKBest(f_classif, k=20)` обучается на **всех** данных и отбирает 20 признаков,
               затем `LogisticRegression()` оценивается кросс-валидацией на отобранных признаках.
            2. **pipeline**: `Pipeline([('select', SelectKBest(f_classif, k=20)), ('model', LogisticRegression())])`
               целиком передаётся в `cross_val_score` — отбор признаков заново обучается внутри каждого фолда.

            ## Подсказки

            - Почему в первом способе accuracy получается намного выше 0.5, хотя связи в данных нет?
            - Любое преобразование, которое «смотрит» на целевую переменную или статистики данных
              (отбор признаков, скейлер, импутер, target encoding), должно обучаться только на обучающей части.

            ## Ожидаемый формат выходного файла

            CSV с колонками `method, accuracy`: две строки — `leaky` и `pipeline` (в этом порядке), accuracy округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.feature_selection import SelectKBest, f_classif
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.pipeline import Pipeline


            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Ошибка: признаки отобраны с учётом целевой переменной ВСЕХ объектов, включая будущие валидационные
                X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
                leaky = cross_val_score(LogisticRegression(), X_sel, y, cv=cv).mean()

                # Правильно: отбор признаков — часть модели и обучается заново в каждом фолде
                pipe = Pipeline([('select', SelectKBest(f_classif, k=20)), ('model', LogisticRegression())])
                honest = cross_val_score(pipe, X, y, cv=cv).mean()

                print(f"с утечкой: {leaky:.3f}   честно: {honest:.3f}   (данные — чистый шум)")
                return pd.DataFrame({'method': ['leaky', 'pipeline'], 'accuracy': [leaky, honest]}).round(4)
        """,
    },
    {
        "name": "concept_drift",
        "title": "Дрейф данных: Population Stability Index",
        "data": data_drift,
        "task": """
            В `input.csv` два набора данных: `sample = reference` (на чём обучали модель) и `sample = current` (что приходит сейчас),
            признаки `age`, `income`, `sessions`. Для каждого признака посчитайте **PSI**:

            1. Границы корзин — децили эталонной выборки: `edges = np.quantile(ref, [0.1, 0.2, ..., 0.9])`
               (9 внутренних границ → 10 корзин; крайние корзины открыты).
            2. Номер корзины значения x: `np.searchsorted(edges, x, side='right')` (от 0 до 9).
            3. Доли объектов в корзинах для reference (pᵢ) и current (qᵢ); нулевые доли замените на 0.0001.
            4. PSI = Σ (qᵢ − pᵢ) · ln(qᵢ / pᵢ).
            5. Статус: `stable` при PSI < 0.1, `moderate` при 0.1 ≤ PSI < 0.25, `drift` при PSI ≥ 0.25.

            ## Подсказки

            - Почему границы берут по эталонной выборке, а не по общей?
            - У дискретного признака `sessions` многие значения совпадают, поэтому некоторые корзины пустые.
              Это нормально, их и заменяем на 0.0001.
            - PSI симметричен по смыслу: большое значение означает, что распределения заметно разошлись в любую сторону.

            ## Ожидаемый формат выходного файла

            CSV с колонками `feature, psi, status` в порядке `age, income, sessions`; psi округлить до 4 знаков.
        """,
        "solution": """
            FEATURES = ['age', 'income', 'sessions']


            def psi(ref, cur, eps=1e-4):
                edges = np.quantile(ref, np.arange(0.1, 1.0, 0.1))
                p = np.bincount(np.searchsorted(edges, ref, side='right'), minlength=10) / len(ref)
                q = np.bincount(np.searchsorted(edges, cur, side='right'), minlength=10) / len(cur)
                p, q = np.maximum(p, eps), np.maximum(q, eps)
                return float(np.sum((q - p) * np.log(q / p)))


            def main():
                df = pd.read_csv('input.csv')
                ref, cur = df[df['sample'] == 'reference'], df[df['sample'] == 'current']
                rows = []
                for f in FEATURES:
                    v = psi(ref[f].to_numpy(), cur[f].to_numpy())
                    status = 'stable' if v < 0.1 else 'moderate' if v < 0.25 else 'drift'
                    rows.append({'feature': f, 'psi': round(v, 4), 'status': status})
                    print(f"{f:>9}: PSI = {v:.4f} → {status}")
                return pd.DataFrame(rows)
        """,
    },
    {
        "name": "shap",
        "title": "Значения Шепли вручную",
        "data": data_shap,
        "task": """
            Модель задана формулой: **f(x) = 3·x₁ + 2·x₂·x₃ − x₃²** (взаимодействие x₂ и x₃ и нелинейность).
            В `background.csv` — фоновая выборка, в `input.csv` — 6 объектов, прогнозы которых нужно объяснить.

            1. Базовая точка b — средние признаков по `background.csv`.
            2. Для подмножества признаков S определим v(S) = f(z), где zᵢ = xᵢ для i ∈ S и zᵢ = bᵢ для остальных.
            3. Значение Шепли признака i — средний вклад по всем подмножествам S ⊆ {1, 2, 3} \\ {i}:
               φᵢ = Σ |S|!·(3 − |S| − 1)! / 3! · [v(S ∪ {i}) − v(S)].

            ## Подсказки

            - Проверьте аддитивность: φ₁ + φ₂ + φ₃ = f(x) − f(b) для каждого объекта.
            - Для линейной части 3·x₁ ответ известен заранее: φ₁ = 3·(x₁ − b₁). Почему вклад x₂·x₃ делится между x₂ и x₃?
            - Перебор 2ⁿ подмножеств реален только для малого n — поэтому библиотека SHAP использует приближения
              и специальные алгоритмы (TreeSHAP).

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, f_x, f_base, phi1, phi2, phi3`; значения округлить до 4 знаков.
        """,
        "solution": """
            from itertools import combinations
            from math import factorial


            def f(x):
                return 3 * x[0] + 2 * x[1] * x[2] - x[2] ** 2


            def shapley(x, base):
                n = len(x)
                phi = np.zeros(n)
                for i in range(n):
                    others = [j for j in range(n) if j != i]
                    for k in range(n):
                        for S in combinations(others, k):
                            weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                            with_i, without_i = base.copy(), base.copy()
                            for j in S:
                                with_i[j] = without_i[j] = x[j]
                            with_i[i] = x[i]
                            phi[i] += weight * (f(with_i) - f(without_i))
                return phi


            def main():
                base = pd.read_csv('background.csv')[['x1', 'x2', 'x3']].mean().to_numpy()
                df = pd.read_csv('input.csv')
                rows = []
                for _, r in df.iterrows():
                    x = r[['x1', 'x2', 'x3']].to_numpy(dtype=float)
                    phi = shapley(x, base)
                    assert np.isclose(phi.sum(), f(x) - f(base))      # аддитивность
                    rows.append({'id': int(r['id']), 'f_x': f(x), 'f_base': f(base), 'phi1': phi[0], 'phi2': phi[1], 'phi3': phi[2]})
                print("аддитивность выполнена для всех объектов")
                return pd.DataFrame(rows).round(4)
        """,
    },
    {
        "name": "tfidf",
        "title": "TF-IDF вручную",
        "data": data_tfidf,
        "task": """
            В `input.csv` — 6 коротких документов (`doc`, `text`). Слова уже в нижнем регистре и разделены пробелами.

            Вычислите TF-IDF так, как это делает `TfidfVectorizer` в scikit-learn по умолчанию:

            1. Токены — слова, разделённые пробелами (`text.split()`).
            2. tf(t, d) — сколько раз слово t встречается в документе d.
            3. idf(t) = ln((1 + N) / (1 + df(t))) + 1, где N — число документов, df(t) — в скольких документах есть t.
            4. Вектор документа tf·idf нормируется по L2 (длина вектора = 1).

            ## Подсказки

            - Слово, которое есть во всех документах, получило бы минимальный idf = 1 — почему «+1» в формуле не даёт ему обнулиться?
            - Сверьтесь: `TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)` должен дать те же числа.
            - Какое слово самое «характерное» для каждого документа?

            ## Ожидаемый формат выходного файла

            CSV в «длинном» формате с колонками `doc, term, tfidf` — только ненулевые значения,
            отсортированные по `doc`, затем по `term` (обычная сортировка строк); tfidf округлить до 4 знаков.
        """,
        "solution": """
            from collections import Counter

            from sklearn.feature_extraction.text import TfidfVectorizer


            def main():
                df = pd.read_csv('input.csv')
                docs = [t.split() for t in df['text']]
                N = len(docs)
                vocab = sorted({w for d in docs for w in d})
                dfreq = {w: sum(w in d for d in docs) for w in vocab}
                idf = {w: np.log((1 + N) / (1 + dfreq[w])) + 1 for w in vocab}

                rows, matrix = [], []
                for doc_id, d in zip(df['doc'], docs):
                    tf = Counter(d)
                    vec = np.array([tf[w] * idf[w] for w in vocab])
                    vec = vec / np.linalg.norm(vec)
                    matrix.append(vec)
                    rows += [{'doc': doc_id, 'term': w, 'tfidf': v} for w, v in zip(vocab, vec) if v > 0]

                # Сверка с библиотекой
                ref = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False).fit(df['text'])
                lib = ref.transform(df['text']).toarray()[:, [ref.vocabulary_[w] for w in vocab]]
                assert np.allclose(lib, np.array(matrix))
                print("совпадает с TfidfVectorizer")

                return pd.DataFrame(rows).sort_values(['doc', 'term']).round(4)
        """,
    },
    {
        "name": "exponential_smoothing",
        "title": "Экспоненциальное сглаживание: SES и Хольт",
        "data": data_smoothing,
        "task": """
            В `input.csv` — ряд из 24 наблюдений (`t` от 0 до 23, `y`). Реализуйте по формулам два метода.

            **Простое сглаживание (SES)**, α = 0.3: уровень l₀ = y₀; для t ≥ 1 прогноз ŷₜ = lₜ₋₁,
            затем lₜ = α·yₜ + (1 − α)·lₜ₋₁. Прогноз вперёд на h шагов: l₂₃ (постоянный).

            **Метод Хольта** (уровень + тренд), α = 0.3, β = 0.2: l₀ = y₀, b₀ = y₁ − y₀; для t ≥ 1 прогноз ŷₜ = lₜ₋₁ + bₜ₋₁,
            затем lₜ = α·yₜ + (1 − α)·(lₜ₋₁ + bₜ₋₁), bₜ = β·(lₜ − lₜ₋₁) + (1 − β)·bₜ₋₁.
            Прогноз вперёд на h шагов: l₂₃ + h·b₂₃.

            ## Подсказки

            - У ряда есть рост — посмотрите, как SES систематически «запаздывает», а Хольт нет.
            - Посчитайте MAE обоих методов по t = 1..23.
            - `statsmodels.tsa.holtwinters.ExponentialSmoothing` с теми же параметрами и инициализацией даст близкие числа.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, y, ses, holt`: строки t = 1..23 — прогнозы на один шаг вперёд (y — фактическое значение),
            затем строки t = 24, 25, 26 — прогнозы в будущее (y пустой). Значения округлить до 4 знаков.
        """,
        "solution": """
            ALPHA, BETA, H = 0.3, 0.2, 3


            def main():
                y = pd.read_csv('input.csv')['y'].to_numpy()
                n = len(y)
                ses_l = y[0]
                holt_l, holt_b = y[0], y[1] - y[0]
                rows = []
                for t in range(1, n):
                    rows.append({'t': t, 'y': y[t], 'ses': ses_l, 'holt': holt_l + holt_b})   # прогнозы до наблюдения y_t
                    ses_l = ALPHA * y[t] + (1 - ALPHA) * ses_l
                    new_l = ALPHA * y[t] + (1 - ALPHA) * (holt_l + holt_b)
                    holt_b = BETA * (new_l - holt_l) + (1 - BETA) * holt_b
                    holt_l = new_l
                for h in range(1, H + 1):
                    rows.append({'t': n - 1 + h, 'y': np.nan, 'ses': ses_l, 'holt': holt_l + h * holt_b})
                result = pd.DataFrame(rows)
                fit = result[result['y'].notna()]
                print(f"MAE: SES {np.abs(fit['y'] - fit['ses']).mean():.2f}   Хольт {np.abs(fit['y'] - fit['holt']).mean():.2f}")
                return result.round(4)
        """,
    },
    {
        "name": "q_learning_sarsa",
        "title": "Q-learning и SARSA по записанной траектории",
        "data": data_trajectories,
        "task": """
            Среда — коридор из 5 клеток (0 — старт, 4 — цель). Действия: 0 — влево, 1 — вправо.
            Шаг стоит −0.1, достижение цели даёт +1 и завершает эпизод.

            В `input.csv` записаны 30 эпизодов агента: `episode, step, s, a, r, s_next, a_next, done`
            (`a_next` — действие, которое агент реально выбрал в `s_next`; −1, если эпизод закончился).

            Начните с нулевых таблиц Q и примените обновления **по порядку строк** (α = 0.5, γ = 0.9):

            1. **Q-learning**: Q[s, a] += α·(r + γ·maxₐ′ Q[s′, a′] − Q[s, a]); при `done = 1` цель просто r.
            2. **SARSA** (отдельная таблица): Q[s, a] += α·(r + γ·Q[s′, a_next] − Q[s, a]); при `done = 1` цель r.

            ## Подсказки

            - Обе таблицы считаются по одним и тем же данным — разница только в цели обновления.
            - Сравните ценности действия «влево» у двух методов: SARSA учитывает, что агент и дальше иногда идёт влево.
            - Q-learning — off-policy: он учит ценности жадной политики, а данные собраны другой (случайной) политикой.

            ## Ожидаемый формат выходного файла

            CSV с колонками `s, q_left, q_right, sarsa_left, sarsa_right` для клеток s = 0..3
            (у терминальной клетки 4 ценности нулевые — её не включаем). Значения округлить до 4 знаков.
        """,
        "solution": """
            ALPHA, GAMMA = 0.5, 0.9


            def main():
                df = pd.read_csv('input.csv')
                q = np.zeros((5, 2))
                sarsa = np.zeros((5, 2))
                for _, r in df.iterrows():
                    s, a, rew, s2, a2, done = int(r['s']), int(r['a']), r['r'], int(r['s_next']), int(r['a_next']), int(r['done'])
                    q_target = rew if done else rew + GAMMA * q[s2].max()               # off-policy: лучшее действие
                    q[s, a] += ALPHA * (q_target - q[s, a])
                    s_target = rew if done else rew + GAMMA * sarsa[s2, a2]             # on-policy: реально выбранное
                    sarsa[s, a] += ALPHA * (s_target - sarsa[s, a])
                print("Q-learning:\\n", q[:4].round(3), "\\nSARSA:\\n", sarsa[:4].round(3))
                return pd.DataFrame({'s': range(4), 'q_left': q[:4, 0], 'q_right': q[:4, 1],
                                     'sarsa_left': sarsa[:4, 0], 'sarsa_right': sarsa[:4, 1]}).round(4)
        """,
    },
]


def main():
    print("Генерация упражнений к Путям 3 и 4:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=301):
        build(exercise, seed)


if __name__ == "__main__":
    main()
