#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор сквозных мини-проектов (projects/<имя>/), по одному на учебный путь.

В отличие от упражнений, у проекта нет единственного правильного ответа: ученик строит свою модель,
делает прогнозы для test.csv и проверяет качество на скрытых метках:
    python tools/check_mini_project.py projects/<имя>

Пороги «базовый / хорошо / отлично» вычисляются здесь же: базовый — по простому бейзлайну,
«отлично» — чуть ниже результата эталонного решения (solution.py), поэтому они заведомо достижимы.

Запуск из корня репозитория:
    python tools/generate_projects.py
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"

CHECK = """
## Как сдать

1. Сохраните прогнозы для `test.csv` в файл `my_predictions.csv` в папке проекта
   (колонки `{id}` и `{pred}` — см. «Формат ответа»).
2. Из корня репозитория выполните:

```bash
python tools/check_mini_project.py projects/{name}
```

Скрипт посчитает метрику на скрытых метках (`answer.csv` — не подглядывайте, иначе проект теряет смысл) и покажет уровень:
**базовый** {base}, **хорошо** {good}, **отлично** {exc} ({direction}).

Эталонное решение с пояснениями — `solution.py`. Откройте его после своей попытки.
"""


# ---------------------------------------------------------------- данные проектов

def churn_data(rng):
    n = 3000
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["month", "one_year", "two_year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.2])
    payment = rng.choice(["card", "bank", "electronic", "mail"], n, p=[0.25, 0.25, 0.35, 0.15])
    monthly = (20 + 25 * (internet == "dsl") + 55 * (internet == "fiber") + rng.normal(0, 8, n)).clip(15, 130).round(2)
    total = (monthly * tenure * rng.uniform(0.9, 1.05, n)).round(2)
    calls = rng.poisson(1.2 + 1.5 * (internet == "fiber"), n)
    age = rng.integers(18, 80, n)
    region = rng.choice([f"R{i}" for i in range(1, 9)], n)
    satisfaction = rng.integers(1, 6, n).astype(float)
    logit = (-1.2 - 0.045 * tenure + 1.4 * (contract == "month") - 0.8 * (contract == "two_year") + 0.9 * (internet == "fiber")
             + 0.6 * (payment == "electronic") + 0.35 * calls - 0.45 * (satisfaction - 3) + 0.012 * (monthly - 65)
             + 0.8 * ((calls >= 3) & (satisfaction <= 2)) + 0.3 * (region == "R3"))
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    total[rng.random(n) < 0.03] = np.nan            # пропуски
    satisfaction[rng.random(n) < 0.12] = np.nan     # не все отвечают на опрос
    df = pd.DataFrame({"id": np.arange(1, n + 1), "tenure": tenure, "monthly_charge": monthly, "total_charges": total,
                       "contract": contract, "internet": internet, "payment": payment, "support_calls": calls, "age": age,
                       "region": region, "satisfaction": satisfaction, "churn": churn})
    return df.iloc[:2000], df.iloc[2000:]


def digits_data(rng):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    df = pd.DataFrame(d.data.astype(int), columns=[f"p{i}" for i in range(64)])
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["label"] = d.target
    return train_test_split(df, test_size=500, stratify=df["label"], random_state=7)


def credit_data(rng):
    n = 9000
    income = rng.lognormal(10.6, 0.5, n).round(0)
    debt_ratio = rng.beta(2, 5, n).round(3)
    age = rng.integers(21, 70, n)
    history = rng.integers(0, 25, n)
    late = rng.poisson(0.4, n)
    utilization = rng.beta(2, 3, n).round(3)
    purpose = rng.choice(["auto", "home", "consumer", "business"], n, p=[0.3, 0.2, 0.4, 0.1])
    logit = (-3.6 + 2.6 * debt_ratio + 0.85 * late + 1.6 * utilization - 0.07 * history - 0.5 * (np.log(income) - 10.6)
             + 0.5 * (purpose == "business") + 0.6 * (late >= 2) * (utilization > 0.6) - 0.01 * (age - 40))
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"id": np.arange(1, n + 1), "income": income, "debt_ratio": debt_ratio, "age": age,
                       "credit_history_years": history, "late_payments": late, "card_utilization": utilization,
                       "purpose": purpose, "default": default})
    return df.iloc[:6000], df.iloc[6000:]


def demand_data(rng):
    days = pd.date_range("2024-01-01", periods=728, freq="D")
    t = np.arange(len(days))
    dow = days.day_of_week.to_numpy()
    doy = days.day_of_year.to_numpy()
    promo = (rng.random(len(days)) < 0.12).astype(int)
    holiday = np.isin(days.strftime("%m-%d"), ["01-01", "01-07", "02-23", "03-08", "05-01", "05-09", "06-12", "11-04", "12-31"]).astype(int)
    base = 200 + 0.12 * t + 25 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    weekly = np.array([0.9, 0.92, 0.95, 1.0, 1.15, 1.35, 1.2])[dow]
    sales = base * weekly * (1 + 0.3 * promo) * (1 + 0.5 * holiday) + rng.normal(0, 12, len(days))
    df = pd.DataFrame({"id": t + 1, "date": days.strftime("%Y-%m-%d"), "promo": promo, "holiday": holiday,
                       "sales": sales.round(0).clip(0)})
    return df.iloc[:-28], df.iloc[-28:]


# ---------------------------------------------------------------- метрики и бейзлайны

def metric(kind, y_true, y_pred, cost=None):
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
    if kind == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    if kind == "accuracy":
        return accuracy_score(y_true, y_pred)
    if kind == "mae":
        return mean_absolute_error(y_true, y_pred)
    if kind == "cost":
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return (cost["fn"] * fn + cost["fp"] * fp) / len(y_true)
    raise ValueError(kind)


def baseline_churn(train, test):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = ["tenure", "monthly_charge"]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression()).fit(train[cols], train["churn"])
    return m.predict_proba(test[cols])[:, 1]


def baseline_digits(train, test):
    from sklearn.neighbors import NearestCentroid
    cols = [f"p{i}" for i in range(64)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NearestCentroid().fit(train[cols], train["label"]).predict(test[cols])


def baseline_credit(train, test):
    return np.zeros(len(test), dtype=int)          # одобряем всех


def baseline_demand(train, test):
    last_week = train["sales"].to_numpy()[-7:]     # сезонный наивный прогноз: «как неделю назад»
    return np.array([last_week[i % 7] for i in range(len(test))])


PROJECT_LIST = [
    {
        "name": "churn_prediction",
        "path": 0,
        "title": "Отток клиентов телеком-оператора",
        "data": churn_data,
        "target": "churn", "pred": "churn_proba", "metric": "roc_auc", "higher": True,
        "baseline": baseline_churn,
        "brief": """
            Телеком-оператор хочет заранее находить клиентов, которые собираются уйти, чтобы предложить им удержание.
            Постройте модель вероятности оттока.

            ## Данные

            - `train.csv` — 2000 клиентов с известным исходом `churn` (1 — ушёл), `test.csv` — 1000 клиентов без исхода.
            - Числовые признаки: `tenure` (месяцев с оператором), `monthly_charge`, `total_charges` (есть пропуски),
              `support_calls`, `age`, `satisfaction` (оценка 1–5 из опроса, много пропусков).
            - Категориальные: `contract`, `internet`, `payment`, `region`.

            ## Шаги (это и есть Путь 1 целиком)

            1. **Разведочный анализ**: распределения, доля оттока, связь признаков с оттоком (шпаргалки по pandas и визуализации).
            2. **Предобработка**: пропуски (подумайте, информативен ли сам факт пропуска в `satisfaction`), кодирование категорий, масштабирование.
            3. **Бейзлайн**: логистическая регрессия; оценка ROC-AUC на кросс-валидации, а не на test.
            4. **Сильная модель**: деревья и градиентный бустинг, подбор гиперпараметров на CV.
            5. **Анализ**: важность признаков — что сильнее всего связано с уходом? Какие рекомендации дать бизнесу?
            6. Сделайте прогноз для `test.csv` и сдайте.

            ## Формат ответа

            `my_predictions.csv` с колонками `id, churn_proba` — вероятность ухода для каждого клиента из `test.csv`.
            Метрика — **ROC-AUC**.
        """,
        "solution": """
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder

            NUM = ['tenure', 'monthly_charge', 'total_charges', 'support_calls', 'age', 'satisfaction']
            CAT = ['contract', 'internet', 'payment', 'region']


            def main():
                train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
                for df in (train, test):
                    df['satisfaction_missing'] = df['satisfaction'].isna().astype(int)   # пропуск в опросе информативен
                    df['charge_per_month'] = df['total_charges'] / df['tenure']
                num = NUM + ['satisfaction_missing', 'charge_per_month']

                # Градиентный бустинг сам обрабатывает пропуски; категории кодируем порядковыми номерами
                prep = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT)],
                                         remainder='passthrough')
                model = Pipeline([('prep', prep), ('gb', HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                    categorical_features=list(range(len(CAT))), random_state=0))])

                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                auc = cross_val_score(model, train[CAT + num], train['churn'], cv=cv, scoring='roc_auc')
                print(f"ROC-AUC на CV: {auc.mean():.3f} ± {auc.std():.3f}")

                model.fit(train[CAT + num], train['churn'])
                return pd.DataFrame({'id': test['id'], 'churn_proba': model.predict_proba(test[CAT + num])[:, 1].round(5)})
        """,
    },
    {
        "name": "digits_recognition",
        "path": 1,
        "title": "Распознавание рукописных цифр нейросетью",
        "data": digits_data,
        "target": "label", "pred": "label", "metric": "accuracy", "higher": True,
        "baseline": baseline_digits,
        "brief": """
            Распознайте рукописные цифры 0–9 по изображениям 8×8 пикселей (классический набор digits из scikit-learn).

            ## Данные

            - `train.csv` — 1297 изображений: `id`, пиксели `p0..p63` (яркость 0–16, по строкам), метка `label`.
            - `test.csv` — 500 изображений без меток.

            ## Шаги (Путь 2)

            1. Посмотрите на данные: выведите несколько цифр как картинки 8×8 (`reshape(8, 8)`).
            2. Нормализуйте пиксели, разделите train на обучение и валидацию.
            3. **MLP**: начните с одного скрытого слоя. Можно написать сеть на NumPy (как в упражнениях Пути 2),
               взять `sklearn.neural_network.MLPClassifier` или PyTorch.
            4. Поэкспериментируйте: число нейронов, функция активации, learning rate, регуляризация (L2, dropout), ранняя остановка.
            5. Бонус: маленькая **CNN** на изображениях 8×8 и аугментация сдвигами на 1 пиксель.
            6. Посмотрите на матрицу ошибок: какие цифры путаются чаще всего и почему?

            ## Формат ответа

            `my_predictions.csv` с колонками `id, label` — предсказанная цифра. Метрика — **accuracy**.
        """,
        "solution": """
            from sklearn.model_selection import train_test_split
            from sklearn.neural_network import MLPClassifier

            COLS = [f'p{i}' for i in range(64)]


            def main():
                train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
                X, Xt = train[COLS] / 16, test[COLS] / 16      # яркость 0–16 → 0–1

                # Ранняя остановка на маленькой выборке останавливает обучение слишком рано (≈0.93) —
                # вместо неё регуляризуем L2 (alpha) и оцениваем качество на отложенной части
                X_tr, X_val, y_tr, y_val = train_test_split(X, train['label'], test_size=0.2, stratify=train['label'], random_state=0)
                mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1000, random_state=0)
                print(f"accuracy на валидации: {mlp.fit(X_tr, y_tr).score(X_val, y_val):.3f}")

                mlp.fit(X, train['label'])          # финальная модель на всех обучающих данных
                return pd.DataFrame({'id': test['id'], 'label': mlp.predict(Xt)})
        """,
    },
    {
        "name": "credit_scoring",
        "path": 2,
        "title": "Кредитный скоринг с ценой ошибок",
        "data": credit_data,
        "target": "default", "pred": "default_pred", "metric": "cost", "higher": False,
        "cost": {"fn": 10, "fp": 1},
        "baseline": baseline_credit,
        "brief": """
            Банк решает, выдавать ли кредит. Невыданный кредит хорошему клиенту — упущенная прибыль (стоимость **1**),
            выданный кредит, который не вернут, — потеря денег (стоимость **10**). Нужно не просто «угадать» дефолт,
            а принять решения с минимальными потерями.

            ## Данные

            - `train.csv` — 6000 заявок с исходом `default` (1 — не вернул), `test.csv` — 3000 заявок без исхода.
            - Признаки: `income`, `debt_ratio`, `age`, `credit_history_years`, `late_payments`, `card_utilization`, `purpose`.
            - Дефолтов немного (несбалансированные классы).

            ## Шаги (Путь 3)

            1. Соберите **пайплайн** предобработки и модели — без утечек (см. упражнение `ml_pipelines`).
            2. Оцените модель на CV: ROC-AUC и PR-AUC. Почему accuracy здесь бесполезна?
            3. **Калибровка**: проверьте, похожи ли предсказанные вероятности на реальные доли дефолтов.
            4. **Выбор порога**: при калиброванных вероятностях оптимальный порог — c_FP / (c_FP + c_FN) = 1/11 ≈ 0.09.
               Проверьте это на валидации, перебрав пороги.
            5. **Интерпретация**: какие признаки сильнее всего влияют на решение (SHAP или permutation importance)?
            6. Сделайте решения для `test.csv` и сдайте.

            ## Формат ответа

            `my_predictions.csv` с колонками `id, default_pred`: 1 — отказать (ожидаем дефолт), 0 — выдать.
            Метрика — **средняя стоимость ошибок на заявку**: (10·FN + 1·FP) / N. Чем меньше, тем лучше.
        """,
        "solution": """
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder

            NUM = ['income', 'debt_ratio', 'age', 'credit_history_years', 'late_payments', 'card_utilization']
            C_FN, C_FP = 10, 1


            def main():
                train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
                prep = ColumnTransformer([('cat', OrdinalEncoder(), ['purpose'])], remainder='passthrough')
                gb = Pipeline([('prep', prep), ('gb', HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=200, max_leaf_nodes=15, categorical_features=[0], random_state=0))])
                # Калибровка вероятностей по кросс-валидации — порог из цены ошибок работает только для честных вероятностей
                model = CalibratedClassifierCV(gb, method='isotonic', cv=5).fit(train[['purpose'] + NUM], train['default'])

                threshold = C_FP / (C_FP + C_FN)        # байесовски оптимальный порог: отказать, если p·C_FN > (1 − p)·C_FP
                proba = model.predict_proba(test[['purpose'] + NUM])[:, 1]
                print(f"порог {threshold:.3f}; доля отказов на test: {(proba >= threshold).mean():.1%}")
                return pd.DataFrame({'id': test['id'], 'default_pred': (proba >= threshold).astype(int)})
        """,
    },
    {
        "name": "demand_forecast",
        "path": 3,
        "title": "Прогноз спроса на 4 недели",
        "data": demand_data,
        "target": "sales", "pred": "sales", "metric": "mae", "higher": False,
        "baseline": baseline_demand,
        "brief": """
            Магазину нужно спланировать закупки: спрогнозируйте ежедневные продажи товара на 28 дней вперёд.

            ## Данные

            - `train.csv` — 700 дней истории: `date`, `promo` (была ли акция), `holiday` (праздник), `sales`.
            - `test.csv` — следующие 28 дней: `date`, `promo`, `holiday` (план акций и календарь известны заранее), без `sales`.

            ## Шаги (Путь 4, специализация «временные ряды»)

            1. Постройте график ряда: тренд, недельная и годовая сезонность, всплески в акции и праздники.
            2. **Бейзлайн**: сезонный наивный прогноз «как неделю назад». Любая модель должна его обогнать.
            3. **Валидация по времени**: отложите последние 28 дней train как валидацию (никакого случайного KFold!).
            4. Модели на выбор: экспоненциальное сглаживание / Хольт-Винтерс, Prophet, градиентный бустинг на признаках
               календаря (день недели, день года, тренд, `promo`, `holiday`) и лагах не ближе 28 дней.
            5. Сравните модели по MAE на валидации, проанализируйте ошибки по дням недели и в дни акций.
            6. Обучите лучшую модель на всём train, сделайте прогноз для `test.csv` и сдайте.

            ## Формат ответа

            `my_predictions.csv` с колонками `id, sales` — прогноз продаж на каждый из 28 дней. Метрика — **MAE** (чем меньше, тем лучше).
        """,
        "solution": """
            from sklearn.ensemble import HistGradientBoostingRegressor


            def features(df, t0):
                d = pd.to_datetime(df['date'])
                return pd.DataFrame({
                    't': (d - t0).dt.days, 'dow': d.dt.dayofweek,
                    'doy_sin': np.sin(2 * np.pi * d.dt.dayofyear / 365.25), 'doy_cos': np.cos(2 * np.pi * d.dt.dayofyear / 365.25),
                    'promo': df['promo'], 'holiday': df['holiday'],
                })


            def main():
                train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
                t0 = pd.to_datetime(train['date']).min()
                X, Xt = features(train, t0), features(test, t0)

                # Деревья плохо экстраполируют тренд: прогнозируем отношение к линейному тренду
                coef = np.polyfit(X['t'], train['sales'], 1)
                trend, trend_t = np.polyval(coef, X['t']), np.polyval(coef, Xt['t'])
                gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[1], random_state=0)

                # Валидация по времени: последние 28 дней train
                cut = len(train) - 28
                gb.fit(X.iloc[:cut], (train['sales'] / trend).iloc[:cut])
                val_pred = gb.predict(X.iloc[cut:]) * trend[cut:]
                naive = train['sales'].to_numpy()[cut - 7:cut]
                naive_pred = np.array([naive[i % 7] for i in range(28)])
                y_val = train['sales'].iloc[cut:].to_numpy()
                print(f"MAE на валидации: бустинг {np.abs(val_pred - y_val).mean():.1f}, сезонный наивный {np.abs(naive_pred - y_val).mean():.1f}")

                gb.fit(X, train['sales'] / trend)
                return pd.DataFrame({'id': test['id'], 'sales': (gb.predict(Xt) * trend_t).round(1)})
        """,
    },
]


SOLUTION_HEADER = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эталонное решение мини-проекта: {title}
Запуск из папки проекта: python solution.py → solution_predictions.csv
"""

import numpy as np
import pandas as pd
'''

SOLUTION_FOOTER = '''

if __name__ == "__main__":
    result = main()
    result.to_csv('solution_predictions.csv', index=False)
    print("✅ Прогнозы сохранены в solution_predictions.csv")
'''


def fmt(v, kind):
    return f"{v:.3f}" if kind in ("roc_auc", "accuracy", "cost") else f"{v:.1f}"


def build(p, seed):
    folder = PROJECTS / p["name"]
    folder.mkdir(parents=True, exist_ok=True)
    train, test = p["data"](np.random.default_rng(seed))
    train.to_csv(folder / "train.csv", index=False)
    test.drop(columns=[p["target"]]).to_csv(folder / "test.csv", index=False)
    test[["id", p["target"]]].to_csv(folder / "answer.csv", index=False)

    solution = SOLUTION_HEADER.format(title=p["title"]) + textwrap.dedent(p["solution"]).rstrip() + "\n" + SOLUTION_FOOTER
    (folder / "solution.py").write_text(solution, encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    run = subprocess.run([sys.executable, "solution.py"], cwd=folder, capture_output=True, text=True, encoding="utf-8", env=env)
    if run.returncode != 0:
        raise SystemExit(run.stderr)
    ref_pred = pd.read_csv(folder / "solution_predictions.csv")
    (folder / "solution_predictions.csv").unlink()

    y = test[p["target"]].to_numpy()
    ref = metric(p["metric"], y, ref_pred[p["pred"]].to_numpy(), p.get("cost"))
    base = metric(p["metric"], y, p["baseline"](train, test), p.get("cost"))
    # «отлично» — немного хуже эталона, «хорошо» — посередине между бейзлайном и «отлично»
    exc = ref - 0.01 * abs(ref) if p["higher"] else ref * 1.03
    good = (base + exc) / 2
    rnd = (lambda v: round(v, 3)) if p["metric"] != "mae" else (lambda v: round(v, 1))
    thresholds = {"baseline": float(rnd(base)), "good": float(rnd(good)), "excellent": float(rnd(exc))}

    config = {"title": p["title"], "path": p["path"], "metric": p["metric"], "higher_is_better": p["higher"],
              "id": "id", "target": p["target"], "prediction": p["pred"], "thresholds": thresholds}
    if p.get("cost"):
        config["cost"] = p["cost"]
    (folder / "project.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    direction = "чем больше, тем лучше" if p["higher"] else "чем меньше, тем лучше"
    readme = f"# Мини-проект: {p['title']}\n\n" + textwrap.dedent(p["brief"]).strip() + "\n"
    readme += CHECK.format(id="id", pred=p["pred"], name=p["name"], base=fmt(thresholds["baseline"], p["metric"]),
                           good=fmt(thresholds["good"], p["metric"]), exc=fmt(thresholds["excellent"], p["metric"]),
                           direction=direction)
    (folder / "README.md").write_text(readme, encoding="utf-8")
    print(f"  ✔ projects/{p['name']}: бейзлайн {fmt(base, p['metric'])}, эталон {fmt(ref, p['metric'])}, пороги {thresholds}")


def main():
    print("Генерация мини-проектов:")
    for seed, p in enumerate(PROJECT_LIST, start=401):
        build(p, seed)


if __name__ == "__main__":
    main()
