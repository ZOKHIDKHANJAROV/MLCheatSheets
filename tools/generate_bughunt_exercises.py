#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений типа 🐞 «Найди ошибку» (bug hunt).

В отличие от обычных упражнений («посчитайте X»), здесь ученик получает рабочий, но неверный
код `buggy.py`: он запускается, печатает подозрительно хорошую (или странную) метрику и пишет
результат в `my_answer.csv`. Задача — найти и исправить ошибки, перезапустить скрипт и пройти
обычную проверку:
    cd exercises/bug_<имя> && python buggy.py
    python tools/check_exercise.py exercises/bug_<имя>

Папки упражнений начинаются с `bug_` — по этому префиксу портал показывает тип «Найди ошибку».
Для каждого упражнения создаются: input.csv, buggy.py, task.md, solution.py (исправленный код
с комментариями «# ИСПРАВЛЕНО: …»), answer.key.

Запуск из корня репозитория:
    python tools/generate_bughunt_exercises.py
"""

import contextlib
import io
import os
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from generate_basic_exercises import EXERCISES, build


def script(text):
    """Текст buggy.py: убираем общий отступ, как у solution."""
    return textwrap.dedent(text).lstrip()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- 1. утечка через масштабирование

def data_scaler_leakage(rng):
    n = 480
    month = np.sort(rng.integers(1, 13, n))
    drift = 1 + 0.1 * (month - 1)                          # доходы и сумма долга растут от месяца к месяцу
    income = (rng.lognormal(4.0, 0.35, n) * drift).round(1)
    debt = (rng.lognormal(3.2, 0.5, n) * drift ** 1.5).round(1)
    age = rng.integers(21, 70, n)
    credit_years = np.clip((age - 21) * rng.uniform(0.1, 0.6, n), 0, None).round(1)
    logit = 5 * (debt / income) - 0.05 * credit_years - 0.02 * (age - 40) - 4.1
    default = rng.binomial(1, sigmoid(logit))
    income[rng.random(n) < 0.12] = np.nan                   # пропуски: клиент не указал доход
    debt[rng.random(n) < 0.084] = np.nan
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "month": month, "income": income, "debt": debt,
        "age": age, "credit_years": credit_years, "default": default,
        "split": np.where(month >= 10, "test", "train"),
    })
    return {"input.csv": df, "buggy.py": script(BUGGY_SCALER)}


BUGGY_SCALER = """
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    FEATURES = ['income', 'debt', 'age', 'credit_years']

    df = pd.read_csv('input.csv')
    X = df[FEATURES]
    y = df['default'].to_numpy()

    # Пропуски заполняем средним, признаки приводим к одному масштабу
    X = SimpleImputer(strategy='mean').fit_transform(X)
    X = StandardScaler().fit_transform(X)

    train = (df['split'] == 'train').to_numpy()
    test = ~train

    model = LogisticRegression(C=0.1)
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[test])[:, 1]
    print(f'Обучение: {train.sum()} заявок, тест: {test.sum()} заявок')
    print(f'ROC-AUC на тесте: {roc_auc_score(y[test], proba):.4f}')

    result = pd.DataFrame({'id': df.loc[test, 'id'].to_numpy(), 'proba': proba.round(4)})
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- 2. утечка целевой переменной

def data_target_leakage(rng):
    n = 600
    income = rng.lognormal(4.1, 0.4, n).round(1)
    loan_amount = (rng.lognormal(5.3, 0.5, n)).round(0)
    term_months = rng.choice([12, 24, 36, 60], n)
    age = rng.integers(21, 70, n)
    late_payments_12m = rng.poisson(0.7, n)
    payment = loan_amount / term_months
    logit = 3.0 * payment / income * 12 / 10 + 0.55 * late_payments_12m - 0.02 * (age - 40) - 2.2
    default = rng.binomial(1, sigmoid(logit + rng.normal(0, 0.8, n)))
    # Признаки, которые появляются только ПОСЛЕ исхода кредита
    collection_amount = np.where(default == 1, loan_amount * rng.uniform(0.2, 0.9, n), 0.0)
    collection_amount[(default == 0) & (rng.random(n) < 0.04)] = 1           # редкие технические взыскания
    collection_amount = collection_amount.round(0)
    days_to_close = np.where(
        default == 1,
        term_months * 30 + rng.integers(90, 720, n),                          # дефолт: закрыт через суд/коллекторов
        (term_months * 30 * rng.uniform(0.4, 1.0, n)).round(0),               # погашен в срок или досрочно
    ).astype(int)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "income": income, "loan_amount": loan_amount,
        "term_months": term_months, "age": age, "late_payments_12m": late_payments_12m,
        "collection_amount": collection_amount, "days_to_close": days_to_close,
        "default": default,
    })
    split = np.array(["train"] * n, dtype=object)
    split[rng.choice(n, n // 4, replace=False)] = "test"
    df["split"] = split
    return {"input.csv": df, "buggy.py": script(BUGGY_TARGET)}


BUGGY_TARGET = """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv('input.csv')

    # Берём все доступные колонки, кроме служебных и целевой
    FEATURES = [c for c in df.columns if c not in ('id', 'default', 'split')]
    print('Признаки:', FEATURES)

    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train['default'])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    print(f'ROC-AUC на тесте: {roc_auc_score(test["default"], proba):.4f}')

    result = pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba.round(4)})
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- 3. отбор признаков вне кросс-валидации

def data_cv_selection(rng):
    n, p = 120, 300
    X = rng.normal(0, 1, (n, p)).round(3)
    logit = 0.7 * X[:, 0] - 0.6 * X[:, 1]                    # сигнал есть только в двух признаках
    target = rng.binomial(1, sigmoid(logit))
    df = pd.DataFrame(X, columns=[f"g{j:03d}" for j in range(1, p + 1)])
    df["target"] = target
    return {"input.csv": df, "buggy.py": script(BUGGY_CV)}


BUGGY_CV = """
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    df = pd.read_csv('input.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    print(f'Объектов: {X.shape[0]}, признаков: {X.shape[1]}')

    # Оставляем 15 самых информативных признаков
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=cv, scoring='accuracy')
    print('Accuracy по фолдам:', scores.round(4))
    print(f'Средняя accuracy: {scores.mean():.4f}')

    result = pd.DataFrame({'fold': range(1, len(scores) + 1), 'accuracy': scores.round(4)})
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- 4. валидация временного ряда

def data_time_series(rng):
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = len(dates)
    t = np.arange(n)
    weekly = np.array([-15, -10, -5, 0, 20, 45, 30])[dates.day_of_week]
    yearly = 25 * np.sin(2 * np.pi * (t - 80) / 365.25)
    noise = np.zeros(n)
    for i in range(1, n):                                    # шум с памятью (AR(1))
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 10)
    sales = 300 + 0.15 * t + weekly + yearly + noise
    return {"input.csv": pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales.round(1)}),
            "buggy.py": script(BUGGY_TS)}


BUGGY_TS = """
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold

    df = pd.read_csv('input.csv', parse_dates=['date'])

    # Признаки: лаги, скользящее среднее за 3 дня, день недели
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_mean_3'] = df['sales'].rolling(3, center=True).mean()
    df['dayofweek'] = df['date'].dt.day_of_week
    df = df.dropna().reset_index(drop=True)

    FEATURES = ['lag_1', 'lag_7', 'rolling_mean_3', 'dayofweek']
    X, y = df[FEATURES], df['sales']

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
        mae = mean_absolute_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        rows.append({'fold': fold, 'mae': round(mae, 4)})
        print(f'Фолд {fold}: MAE = {mae:.4f}')

    result = pd.DataFrame(rows)
    print(f'Средняя MAE: {result["mae"].mean():.4f}')
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- 5. ручной расчёт метрик

def data_metrics(rng):
    n = 200
    y_true = rng.binomial(1, 0.2, n)
    y_score = np.clip(0.3 * y_true + rng.beta(2, 4, n), 0, 1).round(3)
    return {"input.csv": pd.DataFrame({"id": np.arange(1, n + 1), "y_true": y_true, "y_score": y_score}),
            "buggy.py": script(BUGGY_METRICS)}


BUGGY_METRICS = """
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score

    df = pd.read_csv('input.csv')
    y_true = df['y_true']
    y_pred = (df['y_score'] >= 0.5).astype(int)

    # Матрица ошибок для положительного класса (1)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    print(f'TP={tp}, FP={fp}, FN={fn}')

    accuracy = (y_pred == y_true).mean()
    precision = tp / (tp + fn)
    recall = tp / (tp + fp)
    f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred)

    result = pd.DataFrame([{
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
    }]).round(4)
    print(result.to_string(index=False))
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- 6. цикл градиентного спуска

def data_training_loop(rng):
    n = 150
    x1 = rng.normal(0, 1, n).round(3)
    x2 = rng.normal(0, 1, n).round(3)
    y = rng.binomial(1, sigmoid(0.4 + 1.5 * x1 - 1.0 * x2))
    return {"input.csv": pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "buggy.py": script(BUGGY_LOOP)}


BUGGY_LOOP = """
    import numpy as np
    import pandas as pd


    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


    def log_loss(y, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


    df = pd.read_csv('input.csv')
    X = np.column_stack([np.ones(len(df)), df[['x1', 'x2']].to_numpy()])   # столбец единиц — для свободного члена w0
    y = df['y'].to_numpy()
    n = len(y)

    lr = 0.5
    epochs = 30
    w = np.zeros(3)
    grad = np.zeros(3)

    rows = []
    for epoch in range(1, epochs + 1):
        p = sigmoid(X @ w)
        grad += X.T @ (p - y)
        w -= lr * grad
        loss = log_loss(y, sigmoid(X @ w))
        rows.append({'epoch': epoch, 'loss': loss, 'w0': w[0], 'w1': w[1], 'w2': w[2]})
        if epoch % 5 == 0:
            print(f'эпоха {epoch:2d}: loss = {loss:.4f}, w = {np.round(w, 3)}')

    result = pd.DataFrame(rows).round(4)
    result.to_csv('my_answer.csv', index=False)
    print('Результат сохранён в my_answer.csv')
"""


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "bug_scaler_leakage",
        "title": "Найди ошибку: масштабирование до разбиения",
        "data": data_scaler_leakage,
        "task": """
            <!-- шпаргалка: scaling_normalization -->
            Банк обучает скоринговую модель вероятности дефолта. В `input.csv` — 480 заявок за год:
            `income` (доход, есть пропуски), `debt` (текущий долг, есть пропуски), `age`, `credit_years`
            (кредитная история, лет), целевая `default` и колонка `split`: заявки за январь–сентябрь —
            `train`, за октябрь–декабрь — `test` (их модель «ещё не видела»).

            Скрипт `buggy.py` заполняет пропуски средним, стандартизирует признаки, обучает логистическую
            регрессию и сохраняет вероятности дефолта для тестовых заявок в `my_answer.csv`.

            **Симптом.** Скрипт работает, ROC-AUC на тесте выглядит прилично. Но когда ту же модель
            запустили в продакшене на заявки октября **по одной**, вероятности для тех же клиентов
            оказались другими, чем в отчёте. Получается, прогноз для клиента зависит от того, какие ещё
            клиенты попали в тест, — так быть не должно.

            **Ошибок: 2** (одна и та же по природе, в двух местах).

            **Не меняйте:** список признаков, разбиение по `split`, стратегию заполнения (среднее),
            `LogisticRegression(C=0.1)` и порядок строк. Исправлять надо только *то, на каких данных*
            что обучается.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибки, исправьте `buggy.py`
            (или скопируйте его в свой файл), перезапустите и проверьте снова.

            ## Подсказки

            - Представьте, что тестовых заявок ещё не существует — они придут только завтра. Какая строка
              скрипта не смогла бы выполниться в таком виде?
            - У `SimpleImputer` и `StandardScaler` есть «память»: среднее и стандартное отклонение, которые
              они запоминают в `fit`. По каким строкам они сейчас считаются?
            - Ошибка в двух строках с `fit_transform(X)` — оба преобразования «подсмотрели» тестовые заявки.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, proba` — только тестовые заявки в исходном порядке (по возрастанию `id`),
            `proba` — вероятность дефолта, округлённая до 4 знаков.
        """,
        "solution": """
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['income', 'debt', 'age', 'credit_years']


            def main():
                df = pd.read_csv('input.csv')
                train = df[df['split'] == 'train']
                test = df[df['split'] == 'test']

                # ИСПРАВЛЕНО: SimpleImputer и StandardScaler обучаются (fit) только на train —
                # внутри Pipeline это происходит автоматически при model.fit(train).
                # Было: fit_transform на всех 480 строках, т.е. средние/std включали тестовые заявки.
                model = make_pipeline(
                    SimpleImputer(strategy='mean'),
                    StandardScaler(),
                    LogisticRegression(C=0.1),
                )
                model.fit(train[FEATURES], train['default'])

                # ИСПРАВЛЕНО: к тесту применяем только transform с параметрами, выученными на train
                proba = model.predict_proba(test[FEATURES])[:, 1]
                print(f"ROC-AUC на тесте: {roc_auc_score(test['default'], proba):.4f}")

                # Проверка «по одной заявке»: прогноз не зависит от остальных строк теста
                single = model.predict_proba(test[FEATURES].iloc[[0]])[:, 1][0]
                assert np.isclose(single, proba[0])

                return pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba}).round(4)
        """,
        "review": """
            **Ошибка 1.** `SimpleImputer(strategy='mean').fit_transform(X)` на всех данных: пропуски дохода
            и долга заполнены средним, посчитанным в том числе по тестовым заявкам октября–декабря.
            Доходы и долги со временем растут, поэтому «общее» среднее выше, чем среднее по train, и
            тестовые клиенты получают заполнение, которого в продакшене не было бы.

            **Ошибка 2.** `StandardScaler().fit_transform(X)` на всех данных: среднее и std для
            стандартизации тоже включают тест. Модель обучалась на признаках, отмасштабированных с учётом
            будущего, — и прогноз для конкретного клиента зависит от того, кто ещё попал в тестовую выборку.
            По одной заявке так посчитать вообще невозможно.

            Итог в цифрах: ROC-AUC на тесте {auc_bug} с утечкой против {auc_fix} после исправления; вероятности
            отличаются в среднем на {diff_mean} (максимум — на {diff_max}). Метрика почти не поменялась —
            этим такая утечка и коварна: отчёт выглядит нормально, а модель в продакшене ведёт себя иначе.
            На данных с сильным дрейфом или с маленьким train разница бывает гораздо заметнее.

            **Как не попасться:**
            - Сначала разбиение, потом любые `fit`: импутеры, скейлеры, энкодеры, PCA, отбор признаков.
            - Собирайте предобработку и модель в `Pipeline` — тогда `fit` по построению видит только train,
              а та же цепочка корректно работает в кросс-валидации.
            - Тест «по одной заявке»: прогноз для строки не должен меняться от состава остальных строк.

            **Что вынести:** любое преобразование с параметрами (среднее, std, min/max, словарь категорий) —
            это часть модели, и учится оно только на train → шпаргалка «Масштабирование и нормализация».
        """,
    },
    {
        "name": "bug_target_leakage",
        "title": "Найди ошибку: утечка целевой переменной",
        "data": data_target_leakage,
        "task": """
            <!-- шпаргалка: feature_engineering -->
            Модель должна в момент **подачи заявки** на кредит оценить вероятность дефолта. Данные — выгрузка
            из хранилища по уже закрытым договорам (`input.csv`, 600 строк):

            - `income` — доход клиента, тыс. ₽/мес
            - `loan_amount` — сумма кредита, тыс. ₽
            - `term_months` — срок кредита, мес.
            - `age` — возраст клиента
            - `late_payments_12m` — число просрочек по другим кредитам за 12 месяцев
            - `collection_amount` — сумма, переданная на взыскание, тыс. ₽
            - `days_to_close` — сколько дней прошло от выдачи до закрытия договора
            - `default` — целевая переменная (1 — клиент не вернул кредит)
            - `split` — `train` / `test`

            Скрипт `buggy.py` обучает `StandardScaler + LogisticRegression` на всех колонках и сохраняет
            вероятности дефолта для тестовых строк.

            **Симптом.** На тесте ROC-AUC {auc_bug} — модель почти не ошибается. Риск-менеджер радуется,
            модель внедряют… и на новых заявках она работает заметно хуже, а пара признаков на входе
            у новых заявок вообще всегда равна нулю.

            **Ошибок: 2** — два признака, которых в момент подачи заявки ещё не существует.

            **Не меняйте:** модель и её параметры, разбиение по `split`, порядок строк. Новые признаки
            не добавляйте — только уберите лишние.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибки, исправьте код, перезапустите
            и проверьте снова.

            ## Подсказки

            - Для каждого признака задайте вопрос: «знает ли банк это значение в день, когда клиент
              приносит заявку?»
            - Метрика, слишком хорошая для реальной задачи (ROC-AUC 0.99 в кредитном скоринге), — почти
              всегда утечка. Посмотрите, как каждый признак по отдельности связан с `default`
              (например, `df.groupby('default')[признак].mean()`).
            - Подозрительны `collection_amount` и `days_to_close`: когда появляются эти значения — до
              выдачи кредита или после его исхода?

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, proba` — только тестовые строки в исходном порядке, `proba` — вероятность
            дефолта, округлённая до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            # ИСПРАВЛЕНО: убраны collection_amount (взыскание бывает только ПОСЛЕ дефолта)
            # и days_to_close (договор закрывается в конце истории кредита). В момент заявки их нет.
            LEAKY = ['collection_amount', 'days_to_close']
            FEATURES = ['income', 'loan_amount', 'term_months', 'age', 'late_payments_12m']


            def main():
                df = pd.read_csv('input.csv')
                print(df.groupby('default')[LEAKY].mean().round(1))   # видно, что признаки «знают» ответ

                train = df[df['split'] == 'train']
                test = df[df['split'] == 'test']

                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                model.fit(train[FEATURES], train['default'])
                proba = model.predict_proba(test[FEATURES])[:, 1]
                print(f"ROC-AUC на тесте (честный): {roc_auc_score(test['default'], proba):.4f}")

                return pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba}).round(4)
        """,
        "review": """
            **Ошибка 1.** `collection_amount` — сумма, переданная коллекторам. Она больше нуля практически
            только у тех, кто уже ушёл в дефолт, то есть это не причина, а *следствие* целевой переменной.
            У новых заявок она всегда 0.

            **Ошибка 2.** `days_to_close` — длительность договора до закрытия. Нормальные кредиты
            закрываются в срок или досрочно, проблемные — через суд и взыскание, много позже срока.
            Значение становится известным только в конце жизни кредита.

            Итог: ROC-AUC на тесте {auc_bug} с утечкой против {auc_fix} без неё. Честные {auc_fix} — это и есть
            реальное качество модели; {auc_bug} никогда не повторится на новых заявках.

            **Как не попасться:**
            - Для каждого признака фиксируйте *момент*, когда он становится известен, и сравнивайте с
              моментом прогноза. Выгрузки «по закрытым договорам» особенно опасны: в них всё уже известно.
            - Слишком хорошая метрика — повод искать утечку, а не праздновать. Посмотрите на важности
              признаков: один признак, «объясняющий всё», почти всегда подозрителен.
            - Не берите «все колонки, кроме целевой» — список признаков составляется осознанно.

            **Что вынести:** признак допустим, только если он доступен в момент принятия решения →
            шпаргалка «Feature Engineering».
        """,
    },
    {
        "name": "bug_cv_selection",
        "title": "Найди ошибку: отбор признаков до кросс-валидации",
        "data": data_cv_selection,
        "task": """
            <!-- шпаргалка: ml_pipelines -->
            В `input.csv` — 120 образцов и 300 признаков-«генов» `g001…g300` (уровни экспрессии) и бинарная
            целевая `target`. Признаков намного больше, чем объектов, поэтому скрипт `buggy.py` сначала
            оставляет 15 самых информативных (`SelectKBest(f_classif, k=15)`), а затем оценивает
            логистическую регрессию 5-кратной кросс-валидацией и сохраняет accuracy по фолдам.

            **Симптом.** Кросс-валидация показывает среднюю accuracy {acc_bug}. Лаборатория собрала новую
            партию образцов — и на ней модель угадывает заметно хуже. Где-то CV оказалась слишком
            оптимистичной.

            **Ошибок: 1.**

            **Не меняйте:** `k=15`, `f_classif`, `LogisticRegression(max_iter=1000)`, схему
            `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` и метрику accuracy.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибку, исправьте код, перезапустите
            и проверьте снова.

            ## Подсказки

            - Кросс-валидация честна, только если *всё*, что «учится на данных», заново обучается внутри
              каждого фолда — без доступа к его тестовой части.
            - `SelectKBest` смотрит на `y`, чтобы выбрать признаки. На каких строках он это делает?
            - Проблема в том, что `selector.fit_transform(X, y)` выполняется до `cross_val_score` на всех
              120 строках.

            ## Ожидаемый формат выходного файла

            CSV с колонками `fold, accuracy` — 5 строк (фолды 1…5 в порядке `cv.split`), accuracy
            округлена до 4 знаков.
        """,
        "solution": """
            from sklearn.feature_selection import SelectKBest, f_classif
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.pipeline import Pipeline


            def main():
                df = pd.read_csv('input.csv')
                X = df.drop(columns=['target'])
                y = df['target']

                # ИСПРАВЛЕНО: отбор признаков — шаг пайплайна. cross_val_score обучает весь Pipeline
                # заново на train-части каждого фолда, и SelectKBest не видит y тестовой части.
                # Было: selector.fit_transform(X, y) на всех 120 строках до кросс-валидации.
                pipe = Pipeline([
                    ('select', SelectKBest(f_classif, k=15)),
                    ('model', LogisticRegression(max_iter=1000)),
                ])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
                print(f'Честная средняя accuracy: {scores.mean():.4f}')

                return pd.DataFrame({'fold': range(1, len(scores) + 1), 'accuracy': scores}).round(4)
        """,
        "review": """
            **Ошибка.** `SelectKBest` выбирал признаки по всем 120 строкам, включая те, что потом стали
            тестовыми в каждом фолде. Среди 300 случайных признаков всегда найдутся такие, что *случайно*
            коррелируют с `target` на этих конкретных 120 объектах, — и отбор их находит, в том числе
            благодаря тестовым строкам. Дальше кросс-валидация проверяет модель на объектах, по которым
            уже подобраны признаки.

            Итог: средняя accuracy {acc_bug} с утечкой против {acc_fix} в честной схеме. Реальный сигнал в данных
            есть (в двух признаках из 300), но он гораздо слабее, чем казалось.

            **Как не попасться:**
            - Всё, что вызывает `fit` (отбор признаков, скейлер, PCA, target encoding, подбор порога),
              — внутрь `Pipeline`, а в `cross_val_score`/`GridSearchCV` передавайте пайплайн целиком.
            - Чем больше признаков относительно объектов, тем сильнее эффект: при p ≫ n отбор вне CV
              даёт высокую accuracy даже на чистом шуме.
            - Если подбираете гиперпараметры, финальную оценку делайте вложенной CV или на отложенной выборке.

            **Что вынести:** кросс-валидация оценивает *всю процедуру* обучения, включая предобработку, —
            поэтому процедура должна быть одним объектом-пайплайном → шпаргалка «ML Pipelines».
        """,
    },
    {
        "name": "bug_time_series_split",
        "title": "Найди ошибку: валидация временного ряда",
        "data": data_time_series,
        "task": """
            <!-- шпаргалка: time_series_validation -->
            В `input.csv` — ежедневные продажи магазина за 2023–2024 годы (`date`, `sales`). Нужно
            прогнозировать продажи **на завтра**: строка дня t — это прогноз, который делается вечером дня t−1,
            когда известны продажи только по t−1 включительно. Скрипт `buggy.py` строит признаки (лаги,
            скользящее среднее за 3 дня, день недели), обучает `Ridge` и оценивает MAE на 5 фолдах.

            **Симптом.** MAE на валидации около {mae_bug} при типичном дневном разбросе продаж в десятки
            единиц — подозрительно хорошо. В реальной эксплуатации ошибка оказалась более чем вдвое больше.

            **Ошибок: 2** — одна в признаках, одна в схеме валидации.

            **Не меняйте:** набор признаков (`lag_1, lag_7, rolling_mean_3, dayofweek` — смысл каждого
            сохраняется: `rolling_mean_3` — среднее за 3 дня), `Ridge(alpha=1.0)`, число фолдов (5),
            `dropna()` после построения признаков.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибки, исправьте код, перезапустите
            и проверьте снова.

            ## Подсказки

            - В момент прогноза на завтра вы знаете продажи только до сегодняшнего дня. Каждый признак
              строки должен считаться только по *прошлым* дням — и модель должна проверяться только на
              *будущем* относительно обучения.
            - Посмотрите, какие дни входят в окно `rolling(3, center=True)` для строки t. И какие строки
              попадают в train и test при `KFold(shuffle=True)`.
            - Ошибки в строке с `rolling_mean_3` и в выборе схемы `KFold`. Для временных рядов в sklearn
              есть специальный сплиттер.

            ## Ожидаемый формат выходного файла

            CSV с колонками `fold, mae` — 5 строк (фолды 1…5 по порядку), MAE округлена до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import TimeSeriesSplit


            def main():
                df = pd.read_csv('input.csv', parse_dates=['date'])

                df['lag_1'] = df['sales'].shift(1)
                df['lag_7'] = df['sales'].shift(7)
                # ИСПРАВЛЕНО: скользящее среднее только по прошлым 3 дням (t-3 … t-1).
                # Было: rolling(3, center=True) — окно t-1 … t+1 включало сам ответ y[t] и следующий день.
                df['rolling_mean_3'] = df['sales'].shift(1).rolling(3).mean()
                df['dayofweek'] = df['date'].dt.day_of_week
                df = df.dropna().reset_index(drop=True)

                FEATURES = ['lag_1', 'lag_7', 'rolling_mean_3', 'dayofweek']
                X, y = df[FEATURES], df['sales']

                # ИСПРАВЛЕНО: TimeSeriesSplit — обучение всегда на прошлом, проверка на следующем отрезке.
                # Было: KFold(shuffle=True) — модель училась на днях, идущих после тестовых.
                cv = TimeSeriesSplit(n_splits=5)
                rows = []
                for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
                    assert train_idx.max() < test_idx.min()          # прошлое → будущее
                    model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
                    mae = mean_absolute_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
                    rows.append({'fold': fold, 'mae': mae})
                result = pd.DataFrame(rows).round(4)
                print(result, f"\\nСредняя MAE: {result['mae'].mean():.4f}")
                return result
        """,
        "review": """
            **Ошибка 1.** `rolling(3, center=True)` — центрированное окно: для дня t оно усредняет дни
            t−1, t, t+1, то есть содержит сам ответ `sales[t]` (с весом 1/3) и завтрашний день. Вместе с
            `lag_1` модель почти восстанавливает `sales[t] + sales[t+1]` — фактически подсматривает в целевую
            переменную. Правильно — `shift(1).rolling(3)`: окно t−3 … t−1. (`rolling(3)` без `shift` тоже было
            бы утечкой: окно t−2 … t включает сегодняшний ответ.)

            **Ошибка 2.** `KFold(shuffle=True)` перемешивает дни: модель обучается на декабре и проверяется
            на ноябре, а соседние дни (очень похожие из‑за автокорреляции) попадают по разные стороны
            разбиения. Для временного ряда нужен `TimeSeriesSplit`: каждый фолд учится на прошлом и
            проверяется на следующем отрезке.

            Итог: средняя MAE {mae_bug} с ошибками против {mae_fix} после исправления. Честная ошибка
            в {ratio} раза больше — именно её и стоило ждать в эксплуатации.

            **Как не попасться:**
            - Любой признак строки t считайте только из данных ≤ t−1 (или ≤ t, если значение на t
              действительно известно в момент прогноза). Ищите в коде `center=True`, `shift(-k)`,
              `bfill`, нормализацию по всему ряду.
            - Валидируйте «как в жизни»: `TimeSeriesSplit` или expanding/rolling window, при необходимости
              с зазором (`gap`) между train и test.
            - Сравнивайте с наивным прогнозом (`sales[t-1]` или `sales[t-7]`): если модель «на порядок
              лучше наивного», ищите утечку.

            **Что вынести:** на временных рядах и признаки, и валидация обязаны уважать стрелу времени →
            шпаргалка «Валидация временных рядов».
        """,
    },
    {
        "name": "bug_metrics",
        "title": "Найди ошибку: метрики классификации",
        "data": data_metrics,
        "task": """
            <!-- шпаргалка: classification_metrics -->
            В `input.csv` — результаты модели, выявляющей мошеннические транзакции (класс 1 — мошенничество,
            его около 20%): `y_true` — истинный класс, `y_score` — вероятность класса 1 от модели.
            Скрипт `buggy.py` бинаризует прогноз по порогу 0.5 и считает для отчёта accuracy, precision,
            recall, F1 и ROC-AUC.

            **Симптом.** Цифры в отчёте не сходятся со `sklearn.metrics.classification_report` и с
            ROC-кривой, которую построил коллега: у него площадь под кривой заметно больше, а «точность»
            и «полнота» у вас будто поменялись местами.

            **Ошибок: 3.**

            **Не меняйте:** порог 0.5, accuracy (она посчитана верно), формат результата.
            Все метрики — для положительного класса 1.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибки, исправьте код, перезапустите
            и проверьте снова.

            ## Подсказки

            - Сверьте каждую метрику с определением: precision — «какая доля найденных объектов
              действительно класса 1», recall — «какую долю объектов класса 1 мы нашли».
            - ROC-AUC оценивает качество *ранжирования* при всех порогах сразу. Что получится, если
              передать в неё уже бинаризованные метки? F1 для класса 1 — это одно число, а не среднее по классам.
            - Ошибки — в формулах `precision` и `recall`, в аргументе `roc_auc_score` и в параметре
              `average` у `f1_score`.

            ## Ожидаемый формат выходного файла

            CSV с одной строкой и колонками `accuracy, precision, recall, f1, roc_auc`, значения округлены
            до 4 знаков.
        """,
        "solution": """
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


            def main():
                df = pd.read_csv('input.csv')
                y_true = df['y_true']
                y_pred = (df['y_score'] >= 0.5).astype(int)

                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()

                accuracy = (y_pred == y_true).mean()
                # ИСПРАВЛЕНО: precision = TP / (TP + FP) — доля верных среди предсказанных «1»
                precision = tp / (tp + fp)
                # ИСПРАВЛЕНО: recall = TP / (TP + FN) — доля найденных среди настоящих «1»
                recall = tp / (tp + fn)
                # ИСПРАВЛЕНО: F1 для положительного класса — average='binary' (по умолчанию),
                # а не 'macro' (среднее F1 класса 0 и класса 1)
                f1 = f1_score(y_true, y_pred)
                # ИСПРАВЛЕНО: ROC-AUC считается по вероятностям y_score, а не по меткам y_pred
                roc_auc = roc_auc_score(y_true, df['y_score'])

                assert np.isclose(precision, precision_score(y_true, y_pred))
                assert np.isclose(recall, recall_score(y_true, y_pred))
                assert np.isclose(f1, 2 * precision * recall / (precision + recall))

                return pd.DataFrame([{
                    'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
                }]).round(4)
        """,
        "review": """
            **Ошибка 1.** Знаменатели precision и recall перепутаны: было `TP / (TP + FN)` под именем
            precision и `TP / (TP + FP)` под именем recall. Здесь TP = {tp}, FP = {fp}, FN = {fn}, поэтому
            precision {p_bug} → {p_fix}, recall {r_bug} → {r_fix}. Запоминалка: **P**recision — среди
            **P**редсказанных «1», **R**ecall — среди **R**еальных «1».

            **Ошибка 2.** `f1_score(..., average='macro')` — это среднее F1 класса 0 и класса 1. Класс 0
            большой и предсказывается легко, его F1 высокий и «подтягивает» среднее: {f1_bug} вместо
            честных {f1_fix} для класса мошенничества. Для бинарной задачи нужен `average='binary'`
            (значение по умолчанию) — F1 положительного класса.

            **Ошибка 3.** `roc_auc_score(y_true, y_pred)` по меткам 0/1: ROC-«кривая»
            из одной точки, площадь = (TPR + TNR) / 2 = balanced accuracy при пороге 0.5. Получилось
            {auc_bug} вместо {auc_fix} по вероятностям — метрика ранжирования выкинула всю информацию
            о ранжировании.

            **Как не попасться:**
            - Сверяйте ручной расчёт с `sklearn.metrics` (`precision_score`, `recall_score`,
              `classification_report`) — хотя бы `assert np.isclose(...)`.
            - В ROC-AUC, PR-AUC, log loss передавайте вероятности (`predict_proba[:, 1]`), а не `predict`.
            - Всегда явно понимайте, что значит `average`: `binary` — положительный класс, `macro` —
              среднее по классам без весов, `weighted` — с весами по частоте.

            **Что вынести:** метрика — это определение, а не название переменной: проверяйте формулы
            и входы (метки или вероятности) → шпаргалка «Метрики классификации».
        """,
    },
    {
        "name": "bug_training_loop",
        "title": "Найди ошибку: цикл градиентного спуска",
        "data": data_training_loop,
        "task": """
            <!-- шпаргалка: gradient_descent -->
            Скрипт `buggy.py` обучает логистическую регрессию «с нуля» на NumPy полным (batch) градиентным
            спуском: 150 объектов из `input.csv`, признаки `x1, x2` (уже стандартизованы) плюс
            свободный член, целевая `y`. Функция потерь — средняя бинарная кросс-энтропия (log loss):

            L(w) = −(1/n) · Σ [ yᵢ·log pᵢ + (1 − yᵢ)·log(1 − pᵢ) ],  pᵢ = σ(xᵢ·w),
            ∇L(w) = (1/n) · Xᵀ (p − y).

            После каждой эпохи скрипт записывает loss и текущие веса.

            **Симптом.** Loss не убывает плавно, а скачет и к концу обучения растёт, веса раздуваются
            до десятков. Коллега говорит, что `lr = 0.5` для такой задачи — нормальный шаг, и он прав.

            **Ошибок: 2.**

            **Не меняйте:** `lr = 0.5`, `epochs = 30`, нулевую инициализацию весов, формулу loss, порядок
            «шаг → loss после шага» в записи результата.

            Как работать: в папке упражнения запустите `python buggy.py`, проверьте результат
            (раздел «Проверка» ниже) — проверка не пройдёт. Найдите ошибки, исправьте код, перезапустите
            и проверьте снова.

            ## Подсказки

            - Сравните код вычисления градиента с формулой ∇L(w) выше — буква в букву.
            - Градиент — это свойство *текущих* весов. Что происходит с переменной `grad` между эпохами?
            - Обе ошибки — в строке `grad += X.T @ (p - y)` и в том, где `grad` создаётся.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, loss, w0, w1, w2` — 30 строк (эпохи 1…30): loss и веса **после**
            шага каждой эпохи, значения округлены до 4 знаков.
        """,
        "solution": """
            def sigmoid(z):
                return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


            def log_loss(y, p):
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


            def main():
                df = pd.read_csv('input.csv')
                X = np.column_stack([np.ones(len(df)), df[['x1', 'x2']].to_numpy()])
                y = df['y'].to_numpy()
                n = len(y)

                lr = 0.5
                epochs = 30
                w = np.zeros(3)

                rows = []
                for epoch in range(1, epochs + 1):
                    p = sigmoid(X @ w)
                    # ИСПРАВЛЕНО (1): градиент считается заново на каждой эпохе (присваивание, а не +=).
                    # Было: grad = np.zeros(3) до цикла и grad += …, т.е. сумма градиентов всех прошлых эпох.
                    # ИСПРАВЛЕНО (2): деление на n — градиент СРЕДНЕГО loss, как в формуле.
                    # Было: Xᵀ(p − y) без /n — шаг в 150 раз больше задуманного.
                    grad = X.T @ (p - y) / n
                    w -= lr * grad
                    loss = log_loss(y, sigmoid(X @ w))
                    rows.append({'epoch': epoch, 'loss': loss, 'w0': w[0], 'w1': w[1], 'w2': w[2]})

                result = pd.DataFrame(rows).round(4)
                assert result['loss'].is_monotonic_decreasing      # с разумным шагом loss убывает
                print(result.iloc[[0, 9, 19, 29]])
                return result
        """,
        "review": """
            **Ошибка 1.** `grad = np.zeros(3)` создаётся один раз до цикла, а в цикле `grad += …`. На эпохе k
            шаг делается по *сумме* градиентов всех k эпох — это что‑то вроде momentum без затухания: веса
            разгоняются и проскакивают минимум. Градиент — функция текущих весов, его надо вычислять заново
            (`grad = …`), а накопление — осознанная техника (momentum, gradient accumulation по микробатчам
            с последующим обнулением, как `optimizer.zero_grad()` в PyTorch).

            **Ошибка 2.** Нет деления на n: `Xᵀ(p − y)` — градиент *суммы* потерь, а loss в коде — *среднее*.
            Шаг получился в n = 150 раз больше, чем задумано при `lr = 0.5`. Learning rate и нормировка
            градиента связаны: поменяли одно — меняется смысл другого.

            Итог: с ошибками loss после 30 эпох {loss_bug} (минимум по пути {loss_bug_min}, веса доходят до
            {w_bug}), после исправления loss монотонно убывает с {loss_fix_1} до {loss_fix}, веса
            w = [{w_fix}] — близко к тем, с которыми генерировались данные (0.4, 1.5, −1.0).

            **Как не попасться:**
            - Пишите градиент по формуле и проверяйте численно (gradient check: (L(w+ε) − L(w−ε)) / 2ε).
            - Следите за loss: при правильном градиенте и небольшом lr он убывает монотонно. Скачки и рост —
              слишком большой шаг (или неверный градиент).
            - Состояние между итерациями (накопленные градиенты, моменты) должно быть осознанным; всё
              остальное создаётся заново на каждом шаге.

            **Что вынести:** обновление весов — w ← w − lr · ∇L(w) по *текущему* среднему градиенту →
            шпаргалка «Градиентный спуск».
        """,
    },
]


# ---------------------------------------------------------------- цифры для условий и разборов

def run_script(folder, code):
    """Выполняет код (buggy.py или функцию main решения) в папке упражнения и возвращает namespace."""
    answer = Path(folder) / "my_answer.csv"
    saved = answer.read_bytes() if answer.exists() else None      # не затираем ответ ученика
    old = os.getcwd()
    os.chdir(folder)
    try:
        ns = {"__name__": "bughunt"}
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, "buggy.py", "exec"), ns)
        return ns
    finally:
        os.chdir(old)
        if saved is not None:
            answer.write_bytes(saved)
        elif answer.exists():
            answer.unlink()


def fill_numbers(exercise, seed):
    """Подставляет в task/review фактические цифры «до/после», посчитанные на сгенерированных данных."""
    from sklearn.metrics import roc_auc_score
    folder = EXERCISES / exercise["name"]
    folder.mkdir(parents=True, exist_ok=True)
    data = exercise["data"](np.random.default_rng(seed))
    data["input.csv"].to_csv(folder / "input.csv", index=False)
    bug = run_script(folder, data["buggy.py"])
    fix = run_script(folder, "import numpy as np\nimport pandas as pd\n" + textwrap.dedent(exercise["solution"])
                     + "\nRESULT = main()\n")
    df = pd.read_csv(folder / "input.csv")
    name = exercise["name"]
    f = {}
    if name == "bug_scaler_leakage":
        test = df[df["split"] == "test"]
        d = (bug["result"]["proba"].to_numpy() - fix["RESULT"]["proba"].to_numpy())
        f = {"auc_bug": f"{roc_auc_score(test['default'], bug['result']['proba']):.3f}",
             "auc_fix": f"{roc_auc_score(test['default'], fix['RESULT']['proba']):.3f}",
             "diff_mean": f"{np.abs(d).mean():.3f}", "diff_max": f"{np.abs(d).max():.3f}"}
    elif name == "bug_target_leakage":
        test = df[df["split"] == "test"]
        f = {"auc_bug": f"{roc_auc_score(test['default'], bug['result']['proba']):.3f}",
             "auc_fix": f"{roc_auc_score(test['default'], fix['RESULT']['proba']):.3f}"}
    elif name == "bug_cv_selection":
        f = {"acc_bug": f"{bug['result']['accuracy'].mean():.3f}", "acc_fix": f"{fix['RESULT']['accuracy'].mean():.3f}"}
    elif name == "bug_time_series_split":
        b, x = bug["result"]["mae"].mean(), fix["RESULT"]["mae"].mean()
        f = {"mae_bug": f"{b:.1f}", "mae_fix": f"{x:.1f}", "ratio": f"{x / b:.1f}"}
    elif name == "bug_metrics":
        b, x = bug["result"].iloc[0], fix["RESULT"].iloc[0]
        f = {"tp": int(bug["tp"]), "fp": int(bug["fp"]), "fn": int(bug["fn"]),
             "p_bug": f"{b['precision']:.3f}", "p_fix": f"{x['precision']:.3f}",
             "r_bug": f"{b['recall']:.3f}", "r_fix": f"{x['recall']:.3f}",
             "f1_bug": f"{b['f1']:.3f}", "f1_fix": f"{x['f1']:.3f}",
             "auc_bug": f"{b['roc_auc']:.3f}", "auc_fix": f"{x['roc_auc']:.3f}"}
    elif name == "bug_training_loop":
        b, x = bug["result"], fix["RESULT"]
        f = {"loss_bug": f"{b['loss'].iloc[-1]:.3f}", "loss_bug_min": f"{b['loss'].min():.3f}",
             "w_bug": f"{b[['w0', 'w1', 'w2']].abs().to_numpy().max():.0f}",
             "loss_fix_1": f"{x['loss'].iloc[0]:.3f}", "loss_fix": f"{x['loss'].iloc[-1]:.3f}",
             "w_fix": ", ".join(f"{v:.2f}" for v in x[['w0', 'w1', 'w2']].iloc[-1])}
    filled = dict(exercise)
    filled["task"] = exercise["task"].format(**f)
    filled["review"] = exercise["review"].format(**f)
    return filled, f


def main():
    print("Генерация упражнений «Найди ошибку»:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=601):
        filled, numbers = fill_numbers(exercise, seed)
        build(filled, seed)
        print("     ", numbers)


if __name__ == "__main__":
    main()
