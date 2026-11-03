#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по временным рядам и обучению с подкреплением (Путь 4).

Временные ряды: лаговые признаки и walk-forward проверка, разложение «как Prophet» МНК,
окна + скейлер + RNN-резервуар, каузальная дилатированная свёртка (TCN), патчи и каузальное внимание.
RL: итерация по ценности в gridworld, REINFORCE с baseline, GAE и clipped-цель PPO,
цели DQN / Double DQN по мини-батчу из replay buffer.

Только NumPy / pandas / scikit-learn — без statsmodels, prophet, torch и gym.
Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_ts_rl.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build

# ---------------------------------------------------------------- данные


def data_time_series(rng):
    n = 140
    t = np.arange(n)
    weekly = np.array([0.0, -4.0, -3.0, -2.0, 5.0, 14.0, 10.0])   # пн … вс (2024-01-01 — понедельник)
    noise = np.zeros(n)
    for i in range(1, n):                                         # AR(1)-шум: вчерашний «сюрприз» продолжается
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 3)
    sales = 120 + 0.15 * t + weekly[t % 7] + noise
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "sales": sales.round(1)})


def data_prophet(rng):
    ds = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(len(ds), dtype=float)
    trend = 200 + 0.08 * t + 0.12 * np.maximum(0, t - 365)          # излом тренда 2024-01-01
    weekly = 6 * np.sin(2 * np.pi * t / 7) + 3 * np.cos(4 * np.pi * t / 7)
    yearly = 15 * np.sin(2 * np.pi * t / 365.25) - 8 * np.cos(2 * np.pi * t / 365.25)
    holidays = pd.DataFrame([
        *[("new_year", d, -1, 1) for d in ("2023-01-01", "2024-01-01", "2025-01-01")],
        *[("women_day", d, 0, 0) for d in ("2023-03-08", "2024-03-08")],
        *[("victory_day", d, 0, 0) for d in ("2023-05-09", "2024-05-09")],
        *[("unity_day", d, 0, 0) for d in ("2023-11-04", "2024-11-04")],
    ], columns=["holiday", "ds", "lower_window", "upper_window"])
    effect = {"new_year": 40.0, "women_day": 25.0, "victory_day": 15.0, "unity_day": 10.0}
    hol = np.zeros(len(ds))
    for _, h in holidays.iterrows():
        day = pd.Timestamp(h["ds"])
        lo, hi = day + pd.Timedelta(days=h["lower_window"]), day + pd.Timedelta(days=h["upper_window"])
        hol[(ds >= lo) & (ds <= hi)] += effect[h["holiday"]]
    y = trend + weekly + yearly + hol + rng.normal(0, 3, len(ds))
    series = pd.DataFrame({"ds": ds.strftime("%Y-%m-%d"), "y": y.round(2)})
    return {"input.csv": series, "holidays.csv": holidays}


def data_rnn_ts(rng):
    n = 72
    t = np.arange(n)
    y = 100 + 0.8 * t + 12 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    H = 4
    weights = {
        "W_ih": rng.normal(0, 1.0, (H, 1)).round(2).tolist(),   # как nn.RNN.weight_ih_l0: (H × 1)
        "W_hh": rng.normal(0, 0.4, (H, H)).round(2).tolist(),   # weight_hh_l0: (H × H)
        "b_ih": rng.normal(0, 0.2, H).round(2).tolist(),
        "b_hh": rng.normal(0, 0.2, H).round(2).tolist(),
    }
    return {"input.csv": pd.DataFrame({"t": t, "y": y.round(2)}),
            "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_tcn(rng):
    n = 24
    t = np.arange(n)
    x = (2 * np.sin(2 * np.pi * t / 8) + rng.normal(0, 0.5, n)).round(2)
    weights = {}
    for i in (1, 2, 3):
        w = rng.normal(0, 0.6, 3)
        w[0] = abs(w[0]) + 0.3                       # текущий момент — с положительным весом, чтобы ReLU не «умер»
        weights[f"w{i}"] = w.round(2).tolist()
        weights[f"b{i}"] = round(abs(float(rng.normal(0, 0.2))), 2)
    return {"input.csv": pd.DataFrame({"t": np.arange(n), "x": x}),
            "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_patch_attention(rng):
    t = np.arange(48)
    y = 20 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 48)
    d = 4
    weights = {
        "W_emb": rng.normal(0, 0.4, (8, d)).round(2).tolist(),   # патч (8 точек) → вектор d_model = 4
        "b_emb": rng.normal(0, 0.1, d).round(2).tolist(),
        "W_q": rng.normal(0, 0.6, (d, d)).round(2).tolist(),
        "W_k": rng.normal(0, 0.6, (d, d)).round(2).tolist(),
        "W_v": rng.normal(0, 0.6, (d, 2)).round(2).tolist(),
    }
    return {"input.csv": pd.DataFrame({"t": t, "y": y.round(3)}),
            "weights.json": json.dumps(weights, indent=2) + "\n"}


def data_gridworld(rng):
    # Классический мир 3 × 4 (Russell & Norvig): S — старт, # — стена, + — выход (+1), - — яма (−1)
    layout = ["...+",
              ".#.-",
              "S..."]
    rows = [{"row": r, "col": c, "cell": ch} for r, line in enumerate(layout) for c, ch in enumerate(line)]
    return pd.DataFrame(rows)


def softmax_rows(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def data_reinforce(rng):
    theta = rng.normal(0, 0.5, (3, 3)).round(2)
    pi = softmax_rows(theta)
    mean_reward = np.array([[1.0, 0.0, 2.0], [0.5, 1.5, -0.5], [2.0, -1.0, 0.0]])
    rows = []
    for ep in range(1, 11):
        s = 0
        for t in range(6):
            a = int(rng.choice(3, p=pi[s]))
            r = round(float(mean_reward[s, a] + rng.normal(0, 0.5)), 1)
            rows.append({"episode": ep, "t": t, "s": s, "a": a, "r": r})
            if rng.random() < 0.3:
                break
            s = int(rng.integers(0, 3))
    theta_df = pd.DataFrame({"s": range(3), "a0": theta[:, 0], "a1": theta[:, 1], "a2": theta[:, 2]})
    return {"input.csv": pd.DataFrame(rows), "theta.csv": theta_df}


def data_ppo(rng):
    T = 16
    done = np.zeros(T, dtype=int)
    done[[5, 11]] = 1                                      # два эпизода закончились внутри роллаута
    value = rng.normal(1.0, 0.4, T + 1).round(3)           # V(s_t); value[T] — состояние после последнего шага
    next_value = value[1:].copy()                          # V(s_{t+1}); после done это уже старт нового эпизода
    reward = rng.choice([0.0, 0.0, 0.5, 1.0], T)
    reward[done == 1] = rng.choice([-1.0, 2.0], 2)
    p_old = rng.uniform(0.15, 0.85, T)
    logp_old = np.log(p_old)
    logp_new = logp_old + rng.normal(0, 0.25, T)
    return pd.DataFrame({"t": np.arange(T), "reward": reward, "done": done, "value": value[:T],
                         "next_value": next_value, "logp_old": logp_old.round(4), "logp_new": logp_new.round(4)})


def data_dqn(rng):
    n = 40
    s = rng.normal(0, 1, (n, 3)).round(2)
    a = rng.integers(0, 3, n)
    s2 = (s + rng.normal(0, 0.5, (n, 3))).round(2)
    r = (rng.normal(0, 1, n)).round(2)
    done = (rng.random(n) < 0.15).astype(int)
    buffer = pd.DataFrame({"idx": np.arange(n),
                           "s_1": s[:, 0], "s_2": s[:, 1], "s_3": s[:, 2], "a": a, "r": r,
                           "next_1": s2[:, 0], "next_2": s2[:, 1], "next_3": s2[:, 2], "done": done})
    W = rng.normal(0, 0.8, (3, 3))
    b = rng.normal(0, 0.3, 3)
    weights = {
        "W": W.round(2).tolist(), "b": b.round(2).tolist(),                                  # онлайн-сеть θ
        "W_target": (0.5 * W + rng.normal(0, 0.8, (3, 3))).round(2).tolist(),                     # целевая θ⁻ (отстала от онлайн-сети)
        "b_target": (b + rng.normal(0, 0.1, 3)).round(2).tolist(),
    }
    return {"input.csv": buffer, "weights.json": json.dumps(weights, indent=2) + "\n"}


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    # ------------------------------------------------------------ временные ряды
    {
        "name": "time_series",
        "title": "Лаговые признаки без утечки и walk-forward проверка",
        "data": data_time_series,
        "task": """
            В `input.csv` — дневные продажи магазина за 140 дней (`date, sales`, с 2024-01-01).
            У ряда есть тренд, недельная сезонность и «инерция» (вчерашний всплеск частично продолжается сегодня) —
            это авторегрессионная часть, о которой рассказывает лекция про ARIMA. Постройте авторегрессию
            на лаговых признаках и честно проверьте её **скользящим началом прогноза** (walk-forward).

            1. Признаки для дня t (строго из прошлого):
               - `lag_1` = sales(t − 1), `lag_7` = sales(t − 7);
               - `roll_mean_7` = среднее sales(t − 7), …, sales(t − 1) — **без** самого sales(t).
            2. Тест — последние 28 дней, 4 фолда по 7 дней подряд. Для фолда k:
               - обучающая выборка — все дни **до** начала фолда, у которых все три признака определены
                 (первые 7 дней ряда выпадают);
               - модель — `LinearRegression()` на `lag_1, lag_7, roll_mean_7` → `sales`;
               - прогноз на шаг вперёд для 7 дней фолда (признаки берутся из фактических прошлых значений).
            3. Бейзлайн — **сезонный наивный** прогноз: ŷ(t) = sales(t − 7).
            4. Для каждого фолда посчитайте MAE модели и MAE бейзлайна.

            ## Подсказки

            - Случайный train/test здесь запрещён: обучение — только на днях раньше фолда, и модель
              переобучается заново на каждом фолде (обучающая выборка растёт на 7 дней).
            - Скользящее среднее «из прошлого» — это `shift(1)` **перед** `rolling(7)`. Если написать
              `rolling(7).mean()` без сдвига, в признак попадёт сам sales(t) — классическая утечка:
              в момент прогноза такой признак посчитать нельзя.
            - Признаки: `df['sales'].shift(1)`, `.shift(7)`, `.shift(1).rolling(7).mean()`; позиция начала фолда
              k = 0..3 — `len(df) - 28 + 7*k`; train — строки с позицией меньше начала и без NaN;
              сезонный наивный прогноз — это просто колонка `lag_7` тестовых строк.

            ## Ожидаемый формат выходного файла

            CSV с колонками `fold, test_start, train_size, mae_model, mae_snaive` — 4 строки (fold = 1..4 по порядку);
            `test_start` — дата первого дня фолда в формате `YYYY-MM-DD`, `train_size` — число обучающих строк.
            MAE округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LinearRegression

            FEATURES = ['lag_1', 'lag_7', 'roll_mean_7']
            TEST_DAYS, FOLD = 28, 7


            def make_features(df):
                out = df.copy()
                out['lag_1'] = out['sales'].shift(1)
                out['lag_7'] = out['sales'].shift(7)
                out['roll_mean_7'] = out['sales'].shift(1).rolling(7).mean()   # окно t−7…t−1, без sales(t)
                return out


            def main():
                df = pd.read_csv('input.csv')
                data = make_features(df)

                # Проверка на утечку: признаки дня t не должны зависеть от будущего
                cut = 100
                assert np.allclose(make_features(df.iloc[:cut])[FEATURES].iloc[-1], data[FEATURES].iloc[cut - 1])

                first_test = len(data) - TEST_DAYS
                rows = []
                for k in range(TEST_DAYS // FOLD):
                    start = first_test + k * FOLD
                    train = data.iloc[:start].dropna(subset=FEATURES)       # только прошлое
                    test = data.iloc[start:start + FOLD]
                    model = LinearRegression().fit(train[FEATURES], train['sales'])
                    pred = model.predict(test[FEATURES])
                    rows.append({'fold': k + 1, 'test_start': test['date'].iloc[0], 'train_size': len(train),
                                 'mae_model': np.mean(np.abs(test['sales'] - pred)),
                                 'mae_snaive': np.mean(np.abs(test['sales'] - test['lag_7']))})

                result = pd.DataFrame(rows)
                ratio = result['mae_model'].mean() / result['mae_snaive'].mean()
                print(f"Средний MAE: модель {result['mae_model'].mean():.2f}, сезонный наивный {result['mae_snaive'].mean():.2f}"
                      f"  (отношение {ratio:.2f} — меньше 1, значит модель полезна)")
                return result.round(4)
        """,
        "review": """
            **Идея.** Авторегрессия на лагах — это AR-часть ARIMA, собранная руками: значение сегодня ≈ линейная
            комбинация вчерашнего, недельной давности и среднего за неделю. Честная оценка — walk-forward:
            обучаемся только на прошлом, прогнозируем следующий кусок, сдвигаем начало и повторяем. Сезонный
            наивный прогноз «как неделю назад» — обязательная планка; лаговая модель обыгрывает его за счёт
            `lag_1` (инерция ряда) и того, что среднее за неделю подхватывает тренд.

            **Типичные ошибки:**
            - `rolling(7).mean()` без `shift(1)` — в признак попадает sales(t) с весом 1/7, MAE модели становится
              немного лучше, чем возможно на деле. Это утечка: в момент прогноза sales(t) ещё неизвестно. Здесь эффект
              скромный, но добавьте лаги 1…6 — и модель восстановит sales(t) из среднего почти точно.
            - Одна модель, обученная на всех 112 днях до теста (или, хуже, на всём ряде), — числа MAE и `train_size`
              не совпадут; переобучать нужно на каждом фолде.
            - `dropna()` по всей таблице до нарезки или удаление другого числа первых строк — `train_size` на 1–7 меньше.
            - Случайный `train_test_split` — соседние дни оказываются по разные стороны, оценка завышена.

            **Что вынести:** признаки ряда — только из прошлого (`shift` перед любым окном), проверка — по времени
            со скользящим началом, сравнение — с сезонным наивным прогнозом (отношение MAE — по сути MASE).
            В sklearn то же разбиение делает `TimeSeriesSplit(n_splits=4, test_size=7)`. → шпаргалка «ARIMA / SARIMA».
        """,
    },
    {
        "name": "prophet_time_series",
        "title": "Разложение «как Prophet»: тренд с изломом, Фурье и праздники МНК",
        "data": data_prophet,
        "task": """
            Prophet раскладывает ряд в сумму y(t) = g(t) + s(t) + h(t) + ε(t). Библиотеки prophet у нас нет —
            соберите ту же аддитивную модель как линейную регрессию на специально построенных признаках.

            Файлы: `input.csv` — дневной ряд (`ds, y`, с 2023-01-01 по 2024-12-31),
            `holidays.csv` — праздники в формате Prophet (`holiday, ds, lower_window, upper_window`).

            Время t — число дней от первой даты ряда (t = 0 для 2023-01-01). Признаки:

            1. **Тренд g(t)** — кусочно-линейный с одной точкой излома t_c = 2024-01-01:
               признаки `t` и `max(0, t − t_c)` (наклон после излома меняется на коэффициент второго признака).
            2. **Недельная сезонность** — ряд Фурье с P = 7, N = 3: sin(2πkt/7), cos(2πkt/7), k = 1..3.
            3. **Годовая сезонность** — P = 365.25, N = 2: sin(2πkt/365.25), cos(2πkt/365.25), k = 1..2.
            4. **Праздники h(t)** — по одной индикаторной колонке на каждое имя `holiday`: 1, если дата попадает
               в окно [ds + lower_window, ds + upper_window] **какого-либо** вхождения этого праздника, иначе 0.
               (У `new_year` окно −1…+1: 31 декабря, 1 и 2 января.)

            Обучите `LinearRegression()` (свободный член — часть тренда) на датах **до 2024-11-01**,
            предскажите период 2024-11-01 … 2024-12-31 и разложите прогноз на компоненты:

            - `trend` = свободный член + вклад двух признаков тренда;
            - `weekly`, `yearly` — вклад соответствующих Фурье-признаков (сумма коэффициент × признак);
            - `holidays` — вклад праздничных колонок;
            - `yhat` = trend + weekly + yearly + holidays.

            ## Подсказки

            - Это ровно то, что показывает `plot_components` в Prophet: прогноз — сумма слагаемых, и каждое
              можно посчитать отдельно, умножив свою группу признаков на свои коэффициенты.
            - Праздник 2025-01-01 лежит за пределами ряда, но его окно (lower_window = −1) накрывает 2024-12-31 —
              поэтому в таблице праздников есть и будущие даты. Признаки праздников должны быть известны на период прогноза.
            - Соберите матрицу X колонками по группам (тренд / weekly / yearly / праздники), обучите модель на
              `X[ds < '2024-11-01']`, затем `X[cols] @ coef[cols]` для каждой группы; проверьте, что сумма
              компонент совпадает с `model.predict(X)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `ds, trend, weekly, yearly, holidays, yhat` — 61 строка (2024-11-01 … 2024-12-31 по порядку),
            `ds` в формате `YYYY-MM-DD`; числа округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LinearRegression

            CHANGEPOINT, TEST_START = '2024-01-01', '2024-11-01'


            def fourier(t, period, order, prefix):
                cols = {}
                for k in range(1, order + 1):
                    cols[f'{prefix}_sin{k}'] = np.sin(2 * np.pi * k * t / period)
                    cols[f'{prefix}_cos{k}'] = np.cos(2 * np.pi * k * t / period)
                return pd.DataFrame(cols)


            def holiday_features(ds, holidays):
                out = {}
                for name, grp in holidays.groupby('holiday', sort=False):
                    flag = np.zeros(len(ds), dtype=bool)
                    for _, h in grp.iterrows():                       # окно каждого вхождения праздника
                        lo = h['ds'] + pd.Timedelta(days=int(h['lower_window']))
                        hi = h['ds'] + pd.Timedelta(days=int(h['upper_window']))
                        flag |= ((ds >= lo) & (ds <= hi)).to_numpy()
                    out[f'hol_{name}'] = flag.astype(int)
                return pd.DataFrame(out)


            def main():
                df = pd.read_csv('input.csv', parse_dates=['ds'])
                holidays = pd.read_csv('holidays.csv', parse_dates=['ds'])
                t = (df['ds'] - df['ds'].iloc[0]).dt.days.to_numpy(dtype=float)
                t_c = (pd.Timestamp(CHANGEPOINT) - df['ds'].iloc[0]).days

                groups = {
                    'trend': pd.DataFrame({'t': t, 't_after_cp': np.maximum(0, t - t_c)}),   # g(t) с изломом
                    'weekly': fourier(t, 7, 3, 'w'),                                          # s(t), P = 7
                    'yearly': fourier(t, 365.25, 2, 'y'),                                     # s(t), P = 365.25
                    'holidays': holiday_features(df['ds'], holidays),                         # h(t)
                }
                X = pd.concat(groups.values(), axis=1)
                train = (df['ds'] < TEST_START).to_numpy()
                model = LinearRegression().fit(X[train], df.loc[train, 'y'])
                coef = pd.Series(model.coef_, index=X.columns)

                comp = pd.DataFrame({'ds': df['ds'].dt.strftime('%Y-%m-%d')})
                for name, g in groups.items():
                    comp[name] = g.to_numpy() @ coef[g.columns].to_numpy()
                comp['trend'] += model.intercept_                     # свободный член относим к тренду
                comp['yhat'] = comp[['trend', 'weekly', 'yearly', 'holidays']].sum(axis=1)
                assert np.allclose(comp['yhat'], model.predict(X))

                print(f"наклон до излома {coef['t']:.3f}/день, после {coef['t'] + coef['t_after_cp']:.3f}/день")
                print("эффекты праздников:", coef.filter(like='hol_').round(1).to_dict())
                test = comp[~train].reset_index(drop=True)
                mae = np.mean(np.abs(df.loc[~train, 'y'].to_numpy() - test['yhat'].to_numpy()))
                print(f"MAE на тесте: {mae:.2f} (шум ряда ≈ 3)")
                return test.round(4)
        """,
        "review": """
            **Идея.** Prophet — это регрессия на функции времени: кусочно-линейный тренд (признак max(0, t − t_c)
            даёт излом наклона), сезонности рядами Фурье и индикаторы праздников с окнами. Поскольку модель
            аддитивна и линейна по параметрам, её можно оценить МНК и разложить прогноз на компоненты — ровно то,
            что рисует `plot_components`. Настоящий Prophet добавляет априорные распределения (регуляризацию)
            на изломы и сезонности и сам расставляет 25 кандидатов в точки излома.

            **Типичные ошибки:**
            - Время t в другой шкале (номер строки с 1, годы, секунды) — тренд и коэффициенты иные, а Фурье-признаки
              сдвигаются по фазе: компоненты не совпадут, хотя `yhat` может быть близок.
            - Окно праздника учтено только для самой даты, без lower/upper window, или пропущено вхождение 2025-01-01 —
              31 декабря прогноз занижен примерно на 40.
            - Свободный член отнесён не к тренду (или потерян) — `trend` и `yhat` сдвинуты на константу.
            - Без признака излома тренд — одна прямая через два года: наклон 2024 года недооценён, прогноз ноября–декабря
              систематически ниже факта.

            **Что вынести:** «прогноз как сумма понятных частей» — сила Prophet (объяснимость, праздники, пропуски
            в данных) и его слабость (он не смотрит на последние значения ряда). Признаки праздников и регрессоров
            должны быть известны на дату прогноза. → шпаргалка «Prophet и методы временных рядов».
        """,
    },
    {
        "name": "rnn_lstm_time_series",
        "title": "Окна, скейлер по train и прогноз рекуррентной сетью",
        "data": data_rnn_ts,
        "task": """
            В `input.csv` — месячный ряд из 72 точек (`t, y`) с трендом и годовой сезонностью.
            В `weights.json` — веса маленькой рекуррентной сети (Elman RNN, вход 1, скрытое состояние H = 4)
            в раскладке PyTorch `nn.RNN`: `W_ih` (4 × 1), `W_hh` (4 × 4), `b_ih` (4), `b_hh` (4).
            Рекуррентные веса **заморожены**; обучать будем только линейную «голову» поверх последнего
            скрытого состояния (так устроены echo state networks — и это уже настоящий прогноз сетью).

            1. **Разбиение по времени:** train — точки t = 0..47, test — t = 48..71.
            2. **Нормализация:** `MinMaxScaler`, обученный **только на y train-части** (t < 48), применить ко всему ряду.
            3. **Окна:** длина L = 12, горизонт 1. Для каждого t от 12 до 71 вход — масштабированные
               значения t − 12 … t − 1, цель — масштабированное y(t). Окно — обучающее, если t < 48,
               иначе тестовое (тестовые окна могут опираться на историю из train — это прошлое, утечки нет).
            4. **Прямой проход RNN** по каждому окну, h₀ = 0, для 12 шагов:
               hₜ = tanh(W_ih·xₜ + b_ih + W_hh·hₜ₋₁ + b_hh); берём последнее состояние h₁₂.
            5. **Голова:** `LinearRegression()` на h₁₂ обучающих окон → масштабированная цель.
               Предскажите тестовые окна и верните прогноз в исходный масштаб (`inverse_transform`).
            6. **Бейзлайн:** наивный прогноз — последнее значение окна, y(t − 1).

            ## Подсказки

            - Сколько окон получится? Ряд длины 72 и окно 12 дают 60 пар «окно → цель»: 36 обучающих и 24 тестовых.
            - Скейлер, обученный на всём ряде, «знает» максимум теста — это утечка. Здесь, наоборот, тест выйдет
              за пределы [0, 1]: из-за тренда значения в конце ряда больше всего, что видел train.
            - Окна: `X = np.array([z[i-12:i] for i in range(12, 72)])`; для окна цикл
              `h = np.tanh(W_ih[:, 0] * x + b_ih + W_hh @ h + b_hh)` по 12 значениям; голова — регрессия
              матрицы состояний (60 × 4) на цели, потом `scaler.inverse_transform(pred.reshape(-1, 1))`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, y_true, y_naive, y_rnn` — 24 строки для тестовых t = 48..71 по порядку;
            значения округлить до 4 знаков.
        """,
        "solution": """
            import json

            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import MinMaxScaler

            WINDOW, N_TRAIN = 12, 48


            def last_hidden_state(window, w):
                h = np.zeros(w['W_hh'].shape[0])
                for x in window:                                     # читаем окно слева направо
                    h = np.tanh(w['W_ih'][:, 0] * x + w['b_ih'] + w['W_hh'] @ h + w['b_hh'])
                return h


            def main():
                df = pd.read_csv('input.csv')
                y = df['y'].to_numpy(dtype=float)
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

                scaler = MinMaxScaler().fit(y[:N_TRAIN].reshape(-1, 1))   # только train!
                z = scaler.transform(y.reshape(-1, 1)).ravel()

                targets = np.arange(WINDOW, len(y))
                X = np.array([z[i - WINDOW:i] for i in targets])           # (60, 12) → для PyTorch (60, 12, 1)
                print("окна:", X.shape, "| масштабированный тест: max =", z[N_TRAIN:].max().round(3))
                H = np.array([last_hidden_state(x, w) for x in X])         # (60, 4)

                is_train = targets < N_TRAIN
                head = LinearRegression().fit(H[is_train], z[targets[is_train]])
                pred = scaler.inverse_transform(head.predict(H[~is_train]).reshape(-1, 1)).ravel()

                t_test = targets[~is_train]
                result = pd.DataFrame({'t': df['t'].to_numpy()[t_test], 'y_true': y[t_test],
                                       'y_naive': y[t_test - 1], 'y_rnn': pred})
                mae = lambda col: np.mean(np.abs(result['y_true'] - result[col]))
                print(f"MAE наивный: {mae('y_naive'):.2f}   MAE RNN: {mae('y_rnn'):.2f}"
                      f"   MASE = {mae('y_rnn') / mae('y_naive'):.2f}")
                return result.round(4)
        """,
        "review": """
            **Идея.** Подготовка данных для рекуррентной сети: сначала разбиение по времени, потом скейлер по train,
            потом нарезка окон в форму (batch, seq_len, 1). Сеть читает окно по шагам, и последнее скрытое состояние —
            сжатое описание окна, из которого линейная голова делает прогноз. Мы обучили только голову (резервуар
            со случайными рекуррентными весами), но конвейер тот же, что у LSTM в PyTorch.

            **Типичные ошибки:**
            - `MinMaxScaler` обучен на всём ряде — значения z другие, все прогнозы сдвинуты (и это утечка из теста).
            - Окно включает саму цель (`z[i-11:i+1]`) или сдвинуто на единицу — модель «подглядывает», MAE неправдоподобно мал.
            - Забыт `b_hh` или порядок `W_hh @ h` перепутан с `h @ W_hh` — скрытые состояния не совпадают.
            - Прогноз не возвращён в исходный масштаб — числа порядка 1 вместо 140–160.

            **Что вынести:** из-за тренда тестовые значения выходят за диапазон train (z > 1), а tanh-сеть плохо
            экстраполирует — поэтому на рядах с трендом предсказывают приращения или нормализуют каждое окно.
            И всегда сравнивайте с наивным прогнозом: здесь MAE сети ≈ 5.4 против ≈ 4.6 у наивного (MASE ≈ 1.16),
            то есть красивая сеть проигрывает «завтра как сегодня» — MASE ≥ 1 означает, что в таком виде она не нужна. → шпаргалка «RNN/LSTM для временных рядов».
        """,
    },
    {
        "name": "cnn_time_series_tcn",
        "title": "TCN: каузальная дилатированная свёртка и рецептивное поле",
        "data": data_tcn,
        "task": """
            В `input.csv` — ряд из 24 точек (`t, x`), в `weights.json` — веса трёх одноканальных свёрточных слоёв
            с ядром k = 3: `w1, b1`, `w2, b2`, `w3, b3` (веса — списки из 3 чисел, смещения — числа).

            Каузальная дилатированная свёртка (формула из лекции; ряд слева дополняется (k − 1)·d нулями):

            y(t) = b + Σⱼ w[j] · x(t − j·d),  j = 0..2,  x(t′) = 0 при t′ < 0

            То есть `w[0]` умножается на текущий момент, `w[1]` — на d шагов назад, `w[2]` — на 2d шагов назад.

            Сеть (один остаточный блок в миниатюре):

            1. h1 = ReLU(conv(x; w1, b1, d = 1))
            2. h2 = ReLU(conv(h1; w2, b2, d = 2))
            3. out = x + conv(h2; w3, b3, d = 4) — остаточная связь: вход прибавляется к выходу.

            Также посчитайте рецептивное поле стека R = 1 + (k − 1)·(d₁ + d₂ + d₃) и для каждого t — число входных точек
            `n_inputs`, от которых структурно зависит out(t): сколько моментов из окна [t − R + 1, t] реально существуют
            (не уходят левее t = 0).

            ## Подсказки

            - Каузальность проверяется просто: измените последнюю точку x — ни один выход в более ранние моменты
              не должен измениться.
            - R = 1 + 2·(1 + 2 + 4) = 15; поэтому `n_inputs` растёт 1, 2, 3, … в начале ряда и дальше постоянен.
            - `xp = np.concatenate([np.zeros((k-1)*d), x])`, затем для каждого t:
              `b + sum(w[j] * xp[pad + t - j*d] for j in range(3))`. Дополнение нулями делается на входе **каждого** слоя
              (для слоя 2 — нули перед h1, а не ReLU(b1)).

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, h1, h2, out, n_inputs` — 24 строки (t = 0..23 по порядку);
            значения округлить до 4 знаков.
        """,
        "solution": """
            import json

            K = 3


            def causal_conv(x, w, b, d):
                pad = (K - 1) * d
                xp = np.concatenate([np.zeros(pad), x])                # нули только слева — будущее недоступно
                return np.array([b + sum(w[j] * xp[pad + t - j * d] for j in range(K)) for t in range(len(x))])


            def forward(x, p):
                relu = lambda v: np.maximum(v, 0)
                h1 = relu(causal_conv(x, p['w1'], p['b1'], 1))
                h2 = relu(causal_conv(h1, p['w2'], p['b2'], 2))
                out = x + causal_conv(h2, p['w3'], p['b3'], 4)        # остаточная связь
                return h1, h2, out


            def main():
                df = pd.read_csv('input.csv')
                x = df['x'].to_numpy(dtype=float)
                p = json.load(open('weights.json', encoding='utf-8'))
                h1, h2, out = forward(x, p)

                R = 1 + (K - 1) * (1 + 2 + 4)
                n_inputs = np.minimum(np.arange(len(x)) + 1, R)        # начало ряда обрезает окно
                print("рецептивное поле R =", R)

                # Проверка каузальности: меняем последнюю точку — ранние выходы не меняются
                x2 = x.copy(); x2[-1] += 10
                assert np.allclose(forward(x2, p)[2][:-1], out[:-1])

                return pd.DataFrame({'t': df['t'], 'h1': h1, 'h2': h2, 'out': out, 'n_inputs': n_inputs}).round(4)
        """,
        "review": """
            **Идея.** Каузальная свёртка — взвешенная сумма текущего и прошлых значений; левое дополнение нулями
            гарантирует, что выход в момент t не видит будущего. Дилатации 1, 2, 4 расширяют поле зрения
            экспоненциально при линейном росте числа слоёв: R = 1 + (k − 1)·Σd = 15. Остаточная связь
            прибавляет вход к выходу блока — как в ResNet и настоящей TCN.

            **Типичные ошибки:**
            - Симметричное дополнение (`padding="same"`, `np.convolve(..., 'same')`) — выход зависит от будущих точек:
              это утечка, и все значения не совпадут.
            - Перевёрнутое ядро: `w[0]` применён к самому старому моменту. Так считает `nn.Conv1d` в PyTorch
              (это кросс-корреляция: последний вес ядра — на текущем моменте), поэтому при переносе весов
              между формулой и библиотекой ядро разворачивают.
            - Дополнение не на каждом слое или ReLU на последнем слое — расхождения начинаются с первых t.
            - Забыта остаточная связь `x +`.

            **Что вынести:** TCN ничего не знает о событиях старше R шагов — подбирайте глубину и дилатации под
            длину важных зависимостей; каузальность проверяйте тестом «изменил последнюю точку — прошлое не сдвинулось».
            → шпаргалка «CNN для временных рядов (TCN)».
        """,
    },
    {
        "name": "transformers_time_series",
        "title": "Патчи PatchTST и каузальное внимание по ним",
        "data": data_patch_attention,
        "task": """
            В `input.csv` — окно истории из 48 точек (`t, y`). В `weights.json` — веса крошечной «головы внимания»:
            `W_emb` (8 × 4), `b_emb` (4), `W_q`, `W_k` (4 × 4), `W_v` (4 × 2).

            1. **Нормализация окна** (RevIN без обучаемых параметров): z = (y − mean) / std по всем 48 точкам, std с `ddof=0`.
            2. **Патчи:** длина P = 8, шаг S = 4 (патчи перекрываются). Патч i покрывает моменты
               4i … 4i + 7; число патчей N = (48 − 8) / 4 + 1 = 11.
            3. **Эмбеддинг:** E = patches · W_emb + b_emb (11 × 4) — каждый патч становится токеном.
            4. **Самовнимание:** Q = E·W_q, K = E·W_k, V = E·W_v; scores = Q·Kᵀ / √4.
            5. **Каузальная маска:** токен i видит только токены j ≤ i (их патчи заканчиваются не позже его патча);
               для j > i scores = −∞. Веса A = softmax по строке, контекст ctx = A·V (11 × 2).

            ## Подсказки

            - Первая строка внимания тривиальна: токену 0 виден только он сам, его вес равен 1.
            - Маска — верхний треугольник без диагонали: `np.triu(np.ones((N, N), bool), k=1)`.
              Удобно заменить там scores на `-np.inf` и делать softmax с вычитанием максимума строки.
            - Патчи: `np.stack([z[s:s + 8] for s in range(0, 41, 4)])`. Для каждой строки i выведите номер ключа с
              максимальным весом (`argmax`), этот вес, вес «на себя» A[i, i] и две координаты контекста.

            ## Ожидаемый формат выходного файла

            CSV с колонками `patch, t_start, t_end, top_key, top_weight, self_weight, ctx1, ctx2` — 11 строк
            (patch = 0..10 по порядку; `t_start, t_end` — первый и последний момент патча; `top_key` — номер патча).
            Вещественные значения округлить до 4 знаков.
        """,
        "solution": """
            import json

            P, S = 8, 4


            def main():
                y = pd.read_csv('input.csv')['y'].to_numpy(dtype=float)
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

                z = (y - y.mean()) / y.std()                                   # RevIN: нормализация окна (ddof=0)
                starts = np.arange(0, len(z) - P + 1, S)
                patches = np.stack([z[s:s + P] for s in starts])               # (11, 8): токен = кусок ряда
                N, d = len(patches), w['W_q'].shape[0]

                E = patches @ w['W_emb'] + w['b_emb']
                Q, K, V = E @ w['W_q'], E @ w['W_k'], E @ w['W_v']
                scores = Q @ K.T / np.sqrt(d)
                scores[np.triu(np.ones((N, N), dtype=bool), k=1)] = -np.inf   # будущее закрыто
                A = np.exp(scores - scores.max(axis=1, keepdims=True))
                A /= A.sum(axis=1, keepdims=True)
                ctx = A @ V

                assert np.allclose(np.triu(A, k=1), 0) and np.allclose(A.sum(axis=1), 1)
                print("токенов:", N, "вместо", len(z), "точек; пар для внимания:", N * N, "вместо", len(z) ** 2)

                return pd.DataFrame({'patch': np.arange(N), 't_start': starts, 't_end': starts + P - 1,
                                     'top_key': A.argmax(axis=1), 'top_weight': A.max(axis=1),
                                     'self_weight': np.diag(A), 'ctx1': ctx[:, 0], 'ctx2': ctx[:, 1]}).round(4)
        """,
        "review": """
            **Идея.** Одна точка ряда почти не несёт смысла, поэтому PatchTST делает токеном кусок ряда: внимание
            сравнивает локальные формы, а длина последовательности падает в разы (11 токенов вместо 48 точек,
            121 пара вместо 2304). Нормализация окна (RevIN) убирает уровень и масштаб — модель видит форму.
            Каузальная маска нужна, когда каждый токен должен опираться только на прошлое (декодерный режим,
            прогноз по шагам); энкодер PatchTST внутри окна истории смотрит во все стороны — там всё и так прошлое.

            **Типичные ошибки:**
            - Нормализация по `ddof=1` (`pd.Series.std()`) — все эмбеддинги чуть другие.
            - Непересекающиеся патчи (шаг 8) — получится 6 токенов вместо 11.
            - Маска с диагональю (`k=0`) — токен не видит сам себя, а у токена 0 строка целиком −∞ и softmax даёт NaN.
            - Забыто деление на √d или softmax по столбцам — веса не совпадут.

            **Что вынести:** патчи + нормализация окна — главные «трюки» трансформеров для рядов; маска внимания
            определяет, что модели разрешено видеть. И помните про холодный душ DLinear: такую модель всё равно
            сравнивают с линейной по окну и с сезонным наивным прогнозом. → шпаргалка «Transformers для временных рядов».
        """,
    },
    # ------------------------------------------------------------ обучение с подкреплением
    {
        "name": "reinforcement_learning_basics",
        "title": "Gridworld: оценка политики и итерация по ценности",
        "data": data_gridworld,
        "task": """
            `input.csv` описывает мир 3 × 4 (`row, col, cell`): `S` — старт, `.` — пустая клетка, `#` — стена,
            `+` — выход (терминальная, вход в неё даёт +1), `-` — яма (терминальная, вход даёт −1).
            Строка 0 — верхняя.

            **Модель среды (MDP).** Действия U, R, D, L (вверх, вправо, вниз, влево). Агент скользит:
            с вероятностью 0.8 идёт куда хотел, с вероятностью 0.1 — в каждую из двух перпендикулярных сторон
            (для U и D это L и R, для L и R — U и D). Если ход упирается в стену или край — агент остаётся на месте.
            Награда за переход в s′: +1, если s′ — выход; −1, если s′ — яма; иначе −0.04 (и когда остался на месте).
            γ = 0.9, ценность терминальных клеток V = 0.

            1. **Оценка политики** — равновероятная случайная политика (каждое действие с вероятностью 1/4):
               V(s) ← Σₐ ¼ · Σₛ′ P(s′ | s, a)·[R(s′) + γ·V(s′)].
            2. **Итерация по ценности:** V*(s) ← maxₐ Σₛ′ P(s′ | s, a)·[R(s′) + γ·V*(s′)].
            3. Для обоих — синхронные обновления с V = 0, пока max |V_new − V| ≥ 1e-10.
            4. **Жадная политика** по V*: действие с наибольшим Q(s, a); при равенстве — первое в порядке U, R, D, L.

            ## Подсказки

            - Удобно написать две функции: `move(s, a)` — куда приводит ход с учётом стен, и `q_value(V, s, a)` —
              сумма по трём исходам (0.8, 0.1, 0.1). Тогда оценка политики и итерация по ценности отличаются
              одной строкой: среднее по действиям или максимум.
            - Проверьте себя: у случайной политики почти все ценности отрицательные (яма рядом, а штраф за шаг
              копится), у оптимальной — положительные; из клетки под выходом (1, 2) лучшее действие — U.
            - Цикл: `V_new = {s: max(q_value(V, s, a) for a in 'URDL') for s in non_terminal}` (для терминальных V = 0),
              остановка по `max(abs(V_new[s] - V[s]))`. Политика — `max('URDL', key=lambda a: q_value(V, s, a))`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `row, col, v_random, v_star, best_action` — 9 строк для нетерминальных клеток без стены,
            по возрастанию `row`, затем `col`; `best_action` — буква U/R/D/L. Значения округлить до 4 знаков.
        """,
        "solution": """
            MOVES = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
            PERP = {'U': 'LR', 'D': 'LR', 'L': 'UD', 'R': 'UD'}
            GAMMA, STEP_REWARD, TOL = 0.9, -0.04, 1e-10


            def main():
                df = pd.read_csv('input.csv')
                grid = {(r, c): cell for r, c, cell in df[['row', 'col', 'cell']].itertuples(index=False)}
                terminal = {s for s, cell in grid.items() if cell in '+-'}
                states = sorted(s for s, cell in grid.items() if cell not in '#+-')

                def move(s, a):
                    nxt = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
                    return nxt if grid.get(nxt, '#') != '#' else s           # стена или край — остаёмся

                def reward(s2):
                    return {'+': 1.0, '-': -1.0}.get(grid[s2], STEP_REWARD)

                def q_value(V, s, a):
                    outcomes = [(0.8, move(s, a))] + [(0.1, move(s, p)) for p in PERP[a]]
                    return sum(p * (reward(s2) + GAMMA * V[s2]) for p, s2 in outcomes)

                def iterate(backup):
                    V = {s: 0.0 for s in grid if grid[s] != '#'}             # терминальные остаются 0
                    while True:
                        V_new = dict(V)
                        for s in states:
                            V_new[s] = backup([q_value(V, s, a) for a in 'URDL'])
                        delta = max(abs(V_new[s] - V[s]) for s in states)
                        V = V_new
                        if delta < TOL:
                            return V

                v_random = iterate(lambda qs: np.mean(qs))                   # оценка политики π(a|s) = 1/4
                v_star = iterate(max)                                        # уравнение оптимальности Беллмана
                best = {s: max('URDL', key=lambda a: q_value(v_star, s, a)) for s in states}

                result = pd.DataFrame({'row': [s[0] for s in states], 'col': [s[1] for s in states],
                                       'v_random': [v_random[s] for s in states], 'v_star': [v_star[s] for s in states],
                                       'best_action': [best[s] for s in states]})
                for r in range(3):
                    print(' '.join({'#': '#', '+': '+', '-': '-'}.get(grid[(r, c)], best.get((r, c), '?')) for c in range(4)))
                return result.round(4)
        """,
        "review": """
            **Идея.** Если модель среды P(s′ | s, a) и награды известны, ценности находятся без единого эпизода —
            многократным применением уравнения Беллмана. Оценка политики усредняет по действиям политики,
            итерация по ценности берёт максимум; обе сходятся, потому что при γ < 1 шаг Беллмана — сжатие.

            **Типичные ошибки:**
            - Детерминированные ходы вместо 0.8 / 0.1 / 0.1 — ценности заметно выше, а возле ямы меняется политика.
            - Удар о стену «запрещён» (действие пропускается) вместо «остаться на месте» с наградой −0.04.
            - Ненулевая ценность терминальных клеток или γ·V(s′) после входа в выход — двойной учёт награды.
            - Асинхронное обновление «на месте» (Гаусс–Зейдель) сходится к тем же V*, но если остановиться раньше,
              числа будут другие; здесь критерий 1e-10 делает ответ одинаковым — расхождение укажет на ошибку в модели.

            **Что вынести:** ценность случайной политики показывает, как плохо «бродить наугад», а V* и жадная
            политика — что даёт планирование. Безмодельные методы (Q-learning, SARSA, DQN) оценивают те же величины
            по опыту, когда P неизвестна. → шпаргалка «Reinforcement Learning Basics».
        """,
    },
    {
        "name": "policy_gradient",
        "title": "REINFORCE: возвраты, baseline и градиент softmax-политики",
        "data": data_reinforce,
        "task": """
            Табличная softmax-политика: π(a | s) = softmax(θ[s])ₐ, 3 состояния и 3 действия.
            Параметры θ — в `theta.csv` (`s, a0, a1, a2`). В `input.csv` — 10 эпизодов, сыгранных этой политикой:
            `episode, t, s, a, r` (r — награда, полученная после действия a на шаге t; эпизод заканчивается
            после своей последней строки).

            1. **Возвраты** (reward-to-go) для каждого шага, γ = 0.9: G(t) = r(t) + γ·r(t+1) + γ²·r(t+2) + … до конца эпизода.
            2. **Baseline** b — среднее всех G(t) по всем шагам всех эпизодов (константа); преимущество A(t) = G(t) − b.
            3. **∇ log π** для табличного softmax: по строке θ[s] это one_hot(a) − π(· | s), по остальным строкам — 0.
            4. **Оценки градиента** (среднее по эпизодам, т.е. сумма по всем шагам, делённая на 10):
               - REINFORCE: g = (1/10)·Σ ∇ log π(aₜ | sₜ) · G(t);
               - с baseline: g_b = (1/10)·Σ ∇ log π(aₜ | sₜ) · (G(t) − b).
            5. **Шаг подъёма** по g_b с α = 0.5: θ_new = θ + α·g_b, и новая политика π_new = softmax(θ_new[s]).

            ## Подсказки

            - Возврат удобно считать с конца эпизода: G = r + γ·G, начиная с G = 0.
            - Проверка: в каждой строке s сумма градиента по трём действиям равна 0 (сумма one_hot − π равна нулю),
              поэтому сумма π_new по действиям по-прежнему 1.
            - Цикл по строкам эпизодов: `g = -pi[s]; g[a] += 1; grad[s] += g * G; grad_b[s] += g * (G - b)`,
              в конце поделить на число эпизодов. Это градиент **подъёма** (максимизируем J) — знак плюс.

            ## Ожидаемый формат выходного файла

            CSV с колонками `s, a, pi, grad_reinforce, grad_baseline, theta_new, pi_new` — 9 строк
            (все пары s = 0..2, a = 0..2, по возрастанию s, затем a); `pi` — исходная π(a | s).
            Значения округлить до 4 знаков.
        """,
        "solution": """
            GAMMA, LR = 0.9, 0.5


            def softmax(z):
                e = np.exp(z - z.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)


            def main():
                episodes = pd.read_csv('input.csv').sort_values(['episode', 't'])
                theta = pd.read_csv('theta.csv').sort_values('s')[['a0', 'a1', 'a2']].to_numpy()
                pi = softmax(theta)

                # Возвраты reward-to-go: с конца эпизода
                G = np.zeros(len(episodes))
                for _, idx in episodes.groupby('episode').indices.items():
                    g = 0.0
                    for i in idx[::-1]:
                        g = episodes['r'].iloc[i] + GAMMA * g
                        G[i] = g
                baseline = G.mean()
                n_episodes = episodes['episode'].nunique()

                grad, grad_b = np.zeros_like(theta), np.zeros_like(theta)
                for (s, a), g_t in zip(episodes[['s', 'a']].to_numpy(), G):
                    glog = -pi[s].copy()
                    glog[a] += 1                                 # ∇θ[s] log π(a|s) = one_hot(a) − π(·|s)
                    grad[s] += glog * g_t
                    grad_b[s] += glog * (g_t - baseline)
                grad /= n_episodes
                grad_b /= n_episodes
                assert np.allclose(grad_b.sum(axis=1), 0)

                theta_new = theta + LR * grad_b                  # градиентный ПОДЪЁМ
                pi_new = softmax(theta_new)
                print(f"baseline = {baseline:.3f}; π до:\\n{pi.round(3)}\\nπ после:\\n{pi_new.round(3)}")

                s_idx, a_idx = np.divmod(np.arange(9), 3)
                return pd.DataFrame({'s': s_idx, 'a': a_idx, 'pi': pi.ravel(), 'grad_reinforce': grad.ravel(),
                                     'grad_baseline': grad_b.ravel(), 'theta_new': theta_new.ravel(),
                                     'pi_new': pi_new.ravel()}).round(4)
        """,
        "review": """
            **Идея.** Теорема о градиенте политики превращает градиент ожидаемого возврата в среднее по траекториям:
            ∇ log π(a | s) показывает, как сделать действие вероятнее, а возврат — насколько это стоит делать.
            Вычитание константного baseline не меняет ожидаемый градиент (E[∇ log π] = 0), но меняет конкретную
            оценку по 10 эпизодам: действия с возвратом ниже среднего теперь подавляются, а не слабо поощряются.

            **Типичные ошибки:**
            - Возврат всего эпизода G(0) на каждом шаге вместо reward-to-go — действие «наказывается» за награды,
              полученные до него.
            - Дисконтирование не с того конца (G(t) = Σ γᵗ·r) или переход через границу эпизодов.
            - Деление на число шагов (19) вместо числа эпизодов (10) — градиент почти вдвое меньше.
            - Знак минус в шаге (θ − α·g) — так делают с loss = −log π·A в PyTorch, но здесь мы уже считаем градиент
              J, и шаг — подъём.

            **Что вынести:** REINFORCE = (возвраты) × (∇ log π); baseline снижает дисперсию бесплатно, лучший
            baseline — V(s), и тогда множитель — преимущество A(s, a): отсюда actor-critic и PPO.
            → шпаргалка «Policy Gradient методы».
        """,
    },
    {
        "name": "ppo_reinforcement_learning",
        "title": "PPO: преимущества GAE и clipped-цель по шагам роллаута",
        "data": data_ppo,
        "task": """
            В `input.csv` — роллаут из 16 шагов: `t, reward, done, value, next_value, logp_old, logp_new`.
            `value` = V(sₜ) критика, `next_value` = V(sₜ₊₁) — оценка состояния после шага (на шаге с `done = 1`
            это уже стартовое состояние нового эпизода); `logp_old` — log π_old(aₜ | sₜ) при сборе данных,
            `logp_new` — log π_θ(aₜ | sₜ) текущей политики после нескольких шагов оптимизации.
            Последний шаг (t = 15) — не конец эпизода: роллаут просто оборван, ценность после него бутстрапится.

            Параметры: γ = 0.99, λ = 0.95, ε = 0.2.

            1. **TD-ошибка:** δₜ = rₜ + γ·(1 − doneₜ)·V(sₜ₊₁) − V(sₜ).
            2. **GAE** (с конца): Aₜ = δₜ + γλ·(1 − doneₜ)·Aₜ₊₁, A после последнего шага = 0.
            3. **Возврат для критика:** retₜ = Aₜ + V(sₜ).
            4. **Нормализация преимуществ:** A_norm = (A − mean(A)) / (std(A) + 1e-8), std **несмещённое**
               (`ddof=1`, как `torch.std`).
            5. **Отношение вероятностей:** ratio = exp(logp_new − logp_old).
            6. **Clipped-цель на шаге:** L = min(ratio·A_norm, clip(ratio, 1 − ε, 1 + ε)·A_norm).
            7. **Флаг обрезки** `clipped` = 1, если (A_norm > 0 и ratio > 1 + ε) или (A_norm < 0 и ratio < 1 − ε) —
               на таком шаге градиент цели по θ равен нулю; иначе 0.

            ## Подсказки

            - Граница эпизода: на шаге с `done = 1` нельзя ни бутстрапить V(sₜ₊₁), ни тянуть Aₜ₊₁ из следующего эпизода —
              обе связи обрываются множителем (1 − done).
            - Обрезается только **выгода**: при A_norm < 0 и ratio > 1 + ε минимум выберет необрезанное
              ratio·A_norm (штраф полностью учитывается), и `clipped` = 0.
            - Цикл `for t in reversed(range(16))`: `delta = r[t] + g*(1-d[t])*nv[t] - v[t]`,
              `last = delta + g*lam*(1-d[t])*last`; дальше всё векторно: `np.minimum(ratio*A, np.clip(ratio, 0.8, 1.2)*A)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, delta, adv, ret, adv_norm, ratio, l_clip, clipped` — 16 строк (t = 0..15 по порядку);
            `clipped` — 0 или 1; остальные значения округлить до 4 знаков.
        """,
        "solution": """
            GAMMA, LAM, EPS = 0.99, 0.95, 0.2


            def main():
                df = pd.read_csv('input.csv').sort_values('t')
                r, done = df['reward'].to_numpy(), df['done'].to_numpy()
                v, nv = df['value'].to_numpy(), df['next_value'].to_numpy()

                delta, adv, last = np.zeros(len(df)), np.zeros(len(df)), 0.0
                for t in reversed(range(len(df))):
                    nonterminal = 1.0 - done[t]
                    delta[t] = r[t] + GAMMA * nonterminal * nv[t] - v[t]       # TD-ошибка критика
                    last = delta[t] + GAMMA * LAM * nonterminal * last         # GAE не переходит границу эпизода
                    adv[t] = last
                ret = adv + v                                                  # цель для критика

                adv_norm = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)
                ratio = np.exp(df['logp_new'].to_numpy() - df['logp_old'].to_numpy())
                l_clip = np.minimum(ratio * adv_norm, np.clip(ratio, 1 - EPS, 1 + EPS) * adv_norm)
                clipped = ((adv_norm > 0) & (ratio > 1 + EPS)) | ((adv_norm < 0) & (ratio < 1 - EPS))

                print(f"L_CLIP (среднее) = {l_clip.mean():.4f}, без обрезки = {(ratio * adv_norm).mean():.4f}, "
                      f"clip fraction = {clipped.mean():.2f}")
                return pd.DataFrame({'t': df['t'], 'delta': delta, 'adv': adv, 'ret': ret, 'adv_norm': adv_norm,
                                     'ratio': ratio, 'l_clip': l_clip, 'clipped': clipped.astype(int)}).round(4)
        """,
        "review": """
            **Идея.** GAE — экспоненциально взвешенная сумма TD-ошибок критика: λ между 0 (одна TD-ошибка, мало
            дисперсии, больше смещения) и 1 (Монте-Карло возврат минус V). Clipped-цель PPO берёт пессимистичный
            минимум: когда новая политика уже ушла дальше 1 ± ε в «выгодную» сторону, градиент по этому шагу
            обнуляется, а ухудшения учитываются полностью.

            **Типичные ошибки:**
            - Бутстрап через `done` (забыт множитель 1 − done в δ) или протягивание A из следующего эпизода
              (забыт он же в рекурсии) — неверны шаги 0–5 и 6–11.
            - Обнуление бутстрапа на последнем шаге t = 15: роллаут оборван по времени, а не закончен — V(s₁₆) нужно учесть.
            - `std` с `ddof=0` — все `adv_norm` и `l_clip` чуть меньше по модулю (в NumPy по умолчанию ddof=0, в PyTorch — 1).
            - clip без min (просто clip(ratio)·A) — при A < 0 и большом ratio штраф ошибочно обрезается.

            **Что вынести:** доля обрезанных шагов (clip fraction) и приближённая KL — главные индикаторы, что
            политика ушла слишком далеко от π_old за эпохи на одном батче. → шпаргалка «PPO».
        """,
    },
    {
        "name": "dqn",
        "title": "DQN: мини-батч из replay buffer, цели target-сети и Double DQN",
        "data": data_dqn,
        "task": """
            В `input.csv` — буфер воспроизведения из 40 переходов: `idx, s_1..s_3, a, r, next_1..next_3, done`
            (состояние — вектор из трёх чисел, действий 3). В `weights.json` — две линейные Q-сети
            Q(s, ·) = s·W + b (W: 3 × 3, признаки × действия; b: 3): онлайн-сеть `W, b` (θ) и целевая `W_target, b_target` (θ⁻).

            1. **Сэмплирование батча:** `rng = np.random.default_rng(2024)`,
               `batch = rng.choice(40, size=8, replace=False)` — индексы строк буфера в этом порядке.
            2. **Текущая оценка:** q_sa = Q(s, a; θ).
            3. **Цель DQN:** y_dqn = r + γ·(1 − done)·maxₐ′ Q(s′, a′; θ⁻), γ = 0.99.
            4. **Double DQN:** a* = argmaxₐ′ Q(s′, a′; θ) (выбор — онлайн-сетью),
               y_double = r + γ·(1 − done)·Q(s′, a*; θ⁻) (оценка — целевой).
            5. **Потери Хьюбера** (smooth L1, δ = 1) для Double DQN: e = y_double − q_sa;
               ℓ = 0.5·e², если |e| < 1, иначе |e| − 0.5.

            ## Подсказки

            - Порядок строк ответа — порядок сэмплирования, а не возрастание `idx`: от фиксированного сида
              зависит, какие переходы попадут в батч.
            - Проверьте себя: y_dqn ≥ y_double на каждой строке (максимум не меньше любого значения) — в этом и
              состоит переоценка, которую убирает Double DQN; на строках с done = 1 обе цели равны r.
            - Векторно: `Q_next_target = S2 @ W_target + b_target`, `a_star = (S2 @ W + b).argmax(1)`,
              `y_double = r + gamma * (1 - done) * Q_next_target[np.arange(8), a_star]`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `idx, a, q_sa, y_dqn, a_star, y_double, huber` — 8 строк в порядке сэмплирования;
            `idx, a, a_star` — целые, остальные значения округлить до 4 знаков.
        """,
        "solution": """
            import json

            GAMMA, SEED, BATCH = 0.99, 2024, 8


            def huber(e, delta=1.0):
                return np.where(np.abs(e) < delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))


            def main():
                buf = pd.read_csv('input.csv')
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                q_online = lambda S: S @ w['W'] + w['b']
                q_target = lambda S: S @ w['W_target'] + w['b_target']

                rng = np.random.default_rng(SEED)
                batch = buf.iloc[rng.choice(len(buf), size=BATCH, replace=False)]   # случайный мини-батч
                S = batch[['s_1', 's_2', 's_3']].to_numpy()
                S2 = batch[['next_1', 'next_2', 'next_3']].to_numpy()
                a, r, done = batch['a'].to_numpy(), batch['r'].to_numpy(), batch['done'].to_numpy()
                rows = np.arange(BATCH)

                q_sa = q_online(S)[rows, a]                                       # Q(s, a; θ)
                y_dqn = r + GAMMA * (1 - done) * q_target(S2).max(axis=1)         # max по целевой сети
                a_star = q_online(S2).argmax(axis=1)                              # Double: выбор онлайн-сетью…
                y_double = r + GAMMA * (1 - done) * q_target(S2)[rows, a_star]    # …оценка целевой
                assert np.all(y_dqn >= y_double - 1e-12)

                loss = huber(y_double - q_sa)
                print(f"средняя переоценка DQN относительно Double: {(y_dqn - y_double).mean():.4f}; "
                      f"loss батча = {loss.mean():.4f}")
                return pd.DataFrame({'idx': batch['idx'].to_numpy(), 'a': a, 'q_sa': q_sa, 'y_dqn': y_dqn,
                                     'a_star': a_star, 'y_double': y_double, 'huber': loss}).round(4)
        """,
        "review": """
            **Идея.** DQN — это Q-learning с двумя стабилизаторами: случайный мини-батч из буфера разрушает
            корреляции соседних переходов, а замороженная целевая сеть θ⁻ делает цель неподвижной между
            синхронизациями. Double DQN разделяет выбор действия (онлайн-сеть) и его оценку (целевая), поэтому
            максимум по шумным оценкам перестаёт систематически завышать цель.

            **Типичные ошибки:**
            - Цель по онлайн-сети (θ вместо θ⁻) или a* по целевой сети — получится обычный DQN, а не Double.
            - Забыт (1 − done) — терминальные переходы получают бутстрап несуществующего будущего.
            - Другой способ сэмплирования (`rng.integers`, `permutation`, `replace=True`) или сортировка батча по `idx` —
              строки не совпадут.
            - Q = W·s вместо s·W (транспонированные веса) — все Q-значения другие.

            **Что вынести:** цель считается без градиента и по целевой сети; Double DQN — изменение одной строки,
            которое почти всегда помогает; Хьюбер не даёт редким огромным TD-ошибкам взорвать градиенты.
            → шпаргалка «Deep Q-Networks (DQN)».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям: временные ряды и RL:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=781):
        build(exercise, seed)


if __name__ == "__main__":
    main()
