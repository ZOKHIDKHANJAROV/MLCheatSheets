#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений по базовым алгоритмам ML (Путь новичка, этапы 2, 4–6).

Для каждого упражнения создаёт в exercises/<имя>/:
    input.csv     — данные (синтетические, с фиксированным seed)
    task.md       — условие
    solution.py   — эталонное решение с комментариями
    answer.csv    — эталонный ответ (результат запуска solution.py)

Запуск из корня репозитория:
    python tools/generate_basic_exercises.py

Повторный запуск перезаписывает эти файлы теми же данными.
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"

CHECK_SECTION = """
## Проверка

Сохраните свой результат в `my_answer.csv` (в папке упражнения) и выполните из корня репозитория:

```bash
python tools/check_exercise.py exercises/{name}
```

Эталонное решение с пояснениями — `solution.py` (загляните в него только после своей попытки).
"""

SOLUTION_HEADER = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: {title}
"""

import pandas as pd
import numpy as np
'''

SOLUTION_FOOTER = '''

if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
'''


def split_column(n, test_share, rng):
    """Колонка split: train/test в случайном порядке."""
    split = np.array(["train"] * n, dtype=object)
    split[rng.choice(n, int(n * test_share), replace=False)] = "test"
    return split


# ---------------------------------------------------------------- данные

def data_regression_metrics(rng):
    y_true = rng.uniform(50, 500, 30).round(1)
    y_pred = (y_true + rng.normal(0, 25, 30)).round(1)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred})


def data_classification_metrics(rng):
    y_true = rng.binomial(1, 0.4, 40)
    y_score = np.clip(0.35 * y_true + rng.uniform(0, 0.65, 40), 0, 1).round(3)
    y_pred = (y_score >= 0.5).astype(int)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_score": y_score})


def data_linreg(rng):
    n = 120
    area = rng.uniform(30, 150, n).round(1)
    rooms = np.clip((area / 30 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    age = rng.integers(0, 60, n)
    distance = rng.uniform(1, 30, n).round(1)
    price = 1.2 * area + 4 * rooms - 0.5 * age - 2.1 * distance + 40 + rng.normal(0, 8, n)
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "area": area, "rooms": rooms, "age": age,
        "distance": distance, "price": price.round(2), "split": split_column(n, 0.25, rng),
    })


def data_logreg(rng):
    n = 200
    tenure = rng.integers(1, 72, n)
    monthly_charge = rng.uniform(20, 120, n).round(2)
    support_calls = rng.poisson(1.5, n)
    age = rng.integers(18, 75, n)
    logit = -0.05 * tenure + 0.03 * monthly_charge + 0.6 * support_calls - 0.01 * age - 1.2
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "tenure": tenure, "monthly_charge": monthly_charge,
        "support_calls": support_calls, "age": age, "churn": churn,
        "split": split_column(n, 0.25, rng),
    })


def data_decision_trees(rng):
    n = 200
    income = rng.uniform(20, 200, n).round(1)
    debt = rng.uniform(0, 100, n).round(1)
    credit_years = rng.integers(0, 30, n)
    age = rng.integers(21, 70, n)
    score = income / 40 - debt / 25 + credit_years / 10 + rng.normal(0, 0.6, n)
    approved = (score > 1.2).astype(int)
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "income": income, "debt": debt,
        "credit_years": credit_years, "age": age, "approved": approved,
        "split": split_column(n, 0.25, rng),
    })


def data_cross_validation(rng):
    n = 100
    X = rng.normal(0, 1, (n, 4)).round(3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 1.0, n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["y"] = y.round(3)
    return df


def data_gradient_boosting(rng):
    n = 240
    temperature = rng.uniform(-10, 35, n).round(1)
    hour = rng.integers(0, 24, n)
    weekend = rng.binomial(1, 2 / 7, n)
    humidity = rng.uniform(20, 100, n).round(0)
    rides = (
        200 + 8 * temperature - 0.25 * (temperature - 22) ** 2
        + 350 * np.exp(-((hour - 8) ** 2) / 3) * (1 - weekend)
        + 400 * np.exp(-((hour - 18) ** 2) / 4) * (1 - weekend)
        + 150 * weekend * (hour > 10) * (hour < 18)
        - 1.2 * humidity + rng.normal(0, 15, n)
    )
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "temperature": temperature, "hour": hour,
        "weekend": weekend, "humidity": humidity, "rides": np.clip(rides, 0, None).round(0),
        "split": split_column(n, 0.25, rng),
    })


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "regression_metrics",
        "title": "Метрики регрессии",
        "data": data_regression_metrics,
        "task": """
            Дан файл `input.csv` с истинными значениями `y_true` и предсказаниями модели `y_pred` (30 объектов).

            Вычислите метрики качества регрессии:

            1. **MAE** — средняя абсолютная ошибка
            2. **MSE** — средняя квадратичная ошибка
            3. **RMSE** — корень из MSE
            4. **R²** — коэффициент детерминации
            5. **MAPE** — средняя абсолютная процентная ошибка **в процентах** (т.е. умноженная на 100)

            ## Формулы

            - MAE = mean(|y − ŷ|)
            - MSE = mean((y − ŷ)²)
            - RMSE = √MSE
            - R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²
            - MAPE = 100 · mean(|y − ŷ| / |y|)

            ## Подсказки

            - Попробуйте сначала посчитать метрики вручную через NumPy, а потом сверить с `sklearn.metrics`.
            - Подумайте: почему RMSE всегда ≥ MAE? Когда они равны?

            ## Ожидаемый формат выходного файла

            CSV с одной строкой и колонками: `mae, mse, rmse, r2, mape` (округлить до 4 знаков).
        """,
        "solution": """
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


            def main():
                df = pd.read_csv('input.csv')
                y, y_hat = df['y_true'], df['y_pred']

                # Вручную — чтобы понимать формулы
                errors = y - y_hat
                mae = np.mean(np.abs(errors))
                mse = np.mean(errors ** 2)
                rmse = np.sqrt(mse)
                r2 = 1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)
                mape = 100 * np.mean(np.abs(errors) / np.abs(y))

                # Сверка с библиотекой
                assert np.isclose(mae, mean_absolute_error(y, y_hat))
                assert np.isclose(mse, mean_squared_error(y, y_hat))
                assert np.isclose(r2, r2_score(y, y_hat))

                print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  (RMSE ≥ MAE: большие ошибки штрафуются сильнее)")
                return pd.DataFrame([{
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape,
                }]).round(4)
        """,
    },
    {
        "name": "classification_metrics",
        "title": "Метрики классификации",
        "data": data_classification_metrics,
        "task": """
            Дан файл `input.csv` с результатами бинарного классификатора (40 объектов):

            - `y_true` — истинный класс (0 или 1)
            - `y_pred` — предсказанный класс (порог 0.5)
            - `y_score` — предсказанная вероятность класса 1

            Вычислите:

            1. **accuracy** — доля верных ответов
            2. **precision** — точность для класса 1
            3. **recall** — полнота для класса 1
            4. **f1** — F1-мера
            5. **roc_auc** — площадь под ROC-кривой (по `y_score`, а не по `y_pred`!)

            ## Подсказки

            - Сначала постройте матрицу ошибок (TP, FP, FN, TN) и посчитайте метрики по формулам.
            - precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2·P·R / (P + R).
            - Для ROC-AUC нужны вероятности, а не метки классов.

            ## Ожидаемый формат выходного файла

            CSV с одной строкой и колонками: `accuracy, precision, recall, f1, roc_auc` (округлить до 4 знаков).
        """,
        "solution": """
            from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                         precision_score, recall_score, roc_auc_score)


            def main():
                df = pd.read_csv('input.csv')
                y, y_pred, y_score = df['y_true'], df['y_pred'], df['y_score']

                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                print(f"Матрица ошибок: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                assert np.isclose(precision, precision_score(y, y_pred))
                assert np.isclose(recall, recall_score(y, y_pred))

                return pd.DataFrame([{
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score(y, y_pred),
                    'roc_auc': roc_auc_score(y, y_score),  # по вероятностям!
                }]).round(4)
        """,
    },
    {
        "name": "linreg",
        "title": "Линейная регрессия: цена квартиры",
        "data": data_linreg,
        "task": """
            Дан файл `input.csv` с данными о квартирах:

            - `area` — площадь, м², `rooms` — число комнат, `age` — возраст дома, лет,
              `distance` — расстояние до центра, км
            - `price` — цена, тыс. у.е. (целевая переменная)
            - `split` — `train` (обучающая выборка) или `test` (тестовая)

            1. Обучите `LinearRegression` (параметры по умолчанию) на строках `train`,
               признаки: `area, rooms, age, distance`
            2. Сделайте предсказания для строк `test`

            ## Подсказки

            - Не используйте тестовые строки при обучении — это утечка данных.
            - Посмотрите на коэффициенты модели (`model.coef_`): какой признак сильнее всего влияет на цену и с каким знаком?
            - Посчитайте RMSE на тесте — пригодится упражнение `regression_metrics`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, price_pred` для тестовых строк в исходном порядке; `price_pred` округлить до 2 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error

            FEATURES = ['area', 'rooms', 'age', 'distance']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                model = LinearRegression().fit(train[FEATURES], train['price'])
                print("Коэффициенты:")
                for name, coef in zip(FEATURES, model.coef_):
                    print(f"  {name:>9}: {coef:+.3f}")
                print(f"  intercept: {model.intercept_:+.3f}")

                pred = model.predict(test[FEATURES])
                print(f"RMSE на тесте: {np.sqrt(mean_squared_error(test['price'], pred)):.2f}")

                return pd.DataFrame({'id': test['id'], 'price_pred': pred.round(2)})
        """,
    },
    {
        "name": "logreg",
        "title": "Логистическая регрессия: отток клиентов",
        "data": data_logreg,
        "task": """
            Дан файл `input.csv` с данными о клиентах телеком-оператора:

            - `tenure` — сколько месяцев клиент с компанией, `monthly_charge` — ежемесячный платёж,
              `support_calls` — число обращений в поддержку, `age` — возраст
            - `churn` — ушёл ли клиент (1) или остался (0) — целевая переменная
            - `split` — `train` или `test`

            1. Масштабируйте признаки `tenure, monthly_charge, support_calls, age` с помощью `StandardScaler`,
               **обучив его только на train**
            2. Обучите `LogisticRegression(C=1.0, max_iter=1000)` на train
            3. Для test посчитайте вероятность оттока и предсказанный класс (порог 0.5)

            ## Подсказки

            - `scaler.fit_transform(train)`, но `scaler.transform(test)` — почему так?
            - Вероятность класса 1: `model.predict_proba(X)[:, 1]`.
            - Посмотрите на знаки коэффициентов: что увеличивает вероятность ухода?

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, churn_proba, churn_pred` для тестовых строк в исходном порядке;
            `churn_proba` округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['tenure', 'monthly_charge', 'support_calls', 'age']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                # Масштабирование обучаем только на train — иначе утечка данных
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train[FEATURES])
                X_test = scaler.transform(test[FEATURES])

                model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, train['churn'])
                for name, coef in zip(FEATURES, model.coef_[0]):
                    print(f"  {name:>15}: {coef:+.3f}")

                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
                print(f"Accuracy={accuracy_score(test['churn'], pred):.3f}  "
                      f"ROC-AUC={roc_auc_score(test['churn'], proba):.3f}")

                return pd.DataFrame({'id': test['id'], 'churn_proba': proba.round(4), 'churn_pred': pred})
        """,
    },
    {
        "name": "decision_trees",
        "title": "Дерево решений: одобрение кредита",
        "data": data_decision_trees,
        "task": """
            Дан файл `input.csv` с заявками на кредит:

            - `income` — доход, тыс., `debt` — текущий долг, тыс., `credit_years` — кредитная история, лет, `age` — возраст
            - `approved` — одобрен ли кредит (целевая переменная)
            - `split` — `train` или `test`

            1. Обучите `DecisionTreeClassifier(max_depth=3, random_state=42)` на train,
               признаки: `income, debt, credit_years, age`
            2. Предскажите класс для test
            3. Посчитайте важность признаков (`feature_importances_`)

            ## Подсказки

            - Деревьям не нужно масштабирование признаков — почему?
            - Нарисуйте дерево: `sklearn.tree.plot_tree` или `export_text` — правила легко прочитать.
            - Сравните accuracy на train и test при `max_depth=3` и без ограничения глубины: где переобучение?

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, approved_pred` для тестовых строк в исходном порядке.
        """,
        "solution": """
            from sklearn.metrics import accuracy_score
            from sklearn.tree import DecisionTreeClassifier, export_text

            FEATURES = ['income', 'debt', 'credit_years', 'age']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                tree = DecisionTreeClassifier(max_depth=3, random_state=42)
                tree.fit(train[FEATURES], train['approved'])
                print(export_text(tree, feature_names=FEATURES))
                print("Важность признаков:", dict(zip(FEATURES, tree.feature_importances_.round(3))))

                # Для сравнения: дерево без ограничения глубины переобучается
                deep = DecisionTreeClassifier(random_state=42).fit(train[FEATURES], train['approved'])
                for name, m in [('max_depth=3', tree), ('без ограничения', deep)]:
                    print(f"{name:>16}: train={accuracy_score(train['approved'], m.predict(train[FEATURES])):.3f}"
                          f"  test={accuracy_score(test['approved'], m.predict(test[FEATURES])):.3f}")

                return pd.DataFrame({'id': test['id'], 'approved_pred': tree.predict(test[FEATURES])})
        """,
    },
    {
        "name": "cross_validation",
        "title": "Кросс-валидация",
        "data": data_cross_validation,
        "task": """
            Дан файл `input.csv` с признаками `x1, x2, x3, x4` и целевой переменной `y` (100 объектов).

            1. Разбейте данные с помощью `KFold(n_splits=5, shuffle=True, random_state=42)`
            2. На каждом фолде обучите `Ridge(alpha=1.0)` и посчитайте **R²** на валидационной части
            3. Сохраните R² каждого фолда

            ## Подсказки

            - Можно написать цикл по `kf.split(X)` вручную, а можно использовать `cross_val_score` с `cv=kf`.
              Результаты должны совпасть — проверьте!
            - Посчитайте среднее и стандартное отклонение R² по фолдам. Что говорит большой разброс?

            ## Ожидаемый формат выходного файла

            CSV с колонками `fold, r2` — 5 строк, `fold` от 1 до 5, `r2` округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            from sklearn.model_selection import KFold, cross_val_score


            def main():
                df = pd.read_csv('input.csv')
                X, y = df[['x1', 'x2', 'x3', 'x4']], df['y']
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                # Вручную: обучаем на 4 фолдах, проверяем на пятом
                scores = []
                for train_idx, val_idx in kf.split(X):
                    model = Ridge(alpha=1.0).fit(X.iloc[train_idx], y.iloc[train_idx])
                    scores.append(r2_score(y.iloc[val_idx], model.predict(X.iloc[val_idx])))

                # То же самое одной строкой
                assert np.allclose(scores, cross_val_score(Ridge(alpha=1.0), X, y, cv=kf, scoring='r2'))
                print(f"R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

                return pd.DataFrame({'fold': range(1, 6), 'r2': np.round(scores, 4)})
        """,
    },
    {
        "name": "gradient_boosting_regression",
        "title": "Градиентный бустинг: прокат велосипедов",
        "data": data_gradient_boosting,
        "task": """
            Дан файл `input.csv` с данными о прокате велосипедов по часам:

            - `temperature` — температура, °C, `hour` — час дня (0–23), `weekend` — выходной (1) или будний (0),
              `humidity` — влажность, %
            - `rides` — число поездок (целевая переменная)
            - `split` — `train` или `test`

            Зависимость нелинейная: утренний и вечерний пик в будни, комфортная температура около 22 °C.

            1. Обучите `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`
               на train, признаки: `temperature, hour, weekend, humidity`
            2. Предскажите `rides` для test

            ## Подсказки

            - Сравните RMSE бустинга и `LinearRegression` на тесте — насколько выигрывает нелинейная модель?
            - Поэкспериментируйте с `learning_rate` и `n_estimators`: как они связаны?

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, rides_pred` для тестовых строк в исходном порядке; `rides_pred` округлить до 2 знаков.
        """,
        "solution": """
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error

            FEATURES = ['temperature', 'hour', 'weekend', 'humidity']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                               max_depth=3, random_state=42)
                gb.fit(train[FEATURES], train['rides'])
                pred = gb.predict(test[FEATURES])

                # Сравнение с линейной моделью: бустинг ловит нелинейность (пики по часам)
                lin = LinearRegression().fit(train[FEATURES], train['rides'])
                rmse = lambda p: np.sqrt(mean_squared_error(test['rides'], p))
                print(f"RMSE бустинг: {rmse(pred):.1f}   RMSE линейная регрессия: {rmse(lin.predict(test[FEATURES])):.1f}")

                return pd.DataFrame({'id': test['id'], 'rides_pred': pred.round(2)})
        """,
    },
]


def build(exercise, seed):
    folder = EXERCISES / exercise["name"]
    folder.mkdir(parents=True, exist_ok=True)

    exercise["data"](np.random.default_rng(seed)).to_csv(folder / "input.csv", index=False)

    task = f"# Упражнение: {exercise['title']}\n\n## Задание\n\n"
    task += textwrap.dedent(exercise["task"]).strip() + "\n"
    task += CHECK_SECTION.format(name=exercise["name"])
    (folder / "task.md").write_text(task, encoding="utf-8")

    solution = SOLUTION_HEADER.format(title=exercise["title"])
    solution += textwrap.dedent(exercise["solution"]).rstrip() + "\n" + SOLUTION_FOOTER
    (folder / "solution.py").write_text(solution, encoding="utf-8")

    subprocess.run([sys.executable, "solution.py"], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL, env={"PYTHONIOENCODING": "utf-8", **__import__("os").environ})
    shutil.move(folder / "solution_output.csv", folder / "answer.csv")
    print(f"  ✔ exercises/{exercise['name']}")


def main():
    print("Генерация упражнений:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=101):
        build(exercise, seed)


if __name__ == "__main__":
    main()
