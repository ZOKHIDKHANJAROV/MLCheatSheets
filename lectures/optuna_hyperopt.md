# Optuna / Hyperopt

<!-- время: 15 мин -->

> **О чём лекция.** Как на практике подбирают гиперпараметры умнее, чем перебором: библиотеки Optuna и Hyperopt строят модель «хороших» и «плохих» настроек (алгоритм TPE), останавливают бесперспективные пробы досрочно и позволяют описывать пространство поиска обычным кодом.

## Зачем это нужно

Байесовская оптимизация с гауссовским процессом хороша на бумаге, но в реальной задаче пространство гиперпараметров неудобное: learning rate непрерывный и логарифмический, число листьев — целое, тип модели — категория, а параметр `gamma` имеет смысл только для одного типа ядра. Нужен инструмент, который:

- работает со смешанными и условными пространствами;
- выбирает следующие пробы с учётом прошлых;
- умеет прерывать явно плохие пробы, не дожидаясь конца обучения;
- сохраняет историю, параллелится и рисует графики.

**Optuna** (Preferred Networks, 2019) — сегодня самый популярный такой инструмент. **Hyperopt** (2013) — его предшественник с той же основной идеей; он встречается в старом коде и в некоторых платформах.

## Интуиция: два «облака» точек

Алгоритм TPE (Tree-structured Parzen Estimator), который по умолчанию используют обе библиотеки, рассуждает так. После нескольких десятков проб разделим их на «хорошие» (лучшие 10–25%) и «плохие» (остальные). Для каждой группы построим оценку плотности: где в пространстве гиперпараметров обычно лежат хорошие пробы, а где — плохие.

Следующую пробу выбираем там, где **хорошие точки плотные, а плохие — редкие**: максимизируем отношение `l(x) / g(x)`, где `l` — плотность хороших, `g` — плохих. Это эквивалентно максимизации Expected Improvement, но без гауссовского процесса: плотности строятся по каждому параметру отдельно, поэтому TPE легко работает с целыми, категориальными и условными параметрами и с сотнями проб.

## Как это работает в Optuna

Три понятия:

- **study** — вся оптимизация: направление (`minimize` / `maximize`), сэмплер, хранилище результатов;
- **trial** — одна проба: внутри неё гиперпараметры *предлагаются* вызовами `trial.suggest_*`;
- **objective** — ваша функция: принимает trial, обучает модель, возвращает метрику.

Главная особенность — пространство поиска описывается **кодом прямо внутри objective** (define-by-run). Можно писать `if`, циклы, выбирать число слоёв и затем размер каждого:

```python
def objective(trial):
    model_type = trial.suggest_categorical("model", ["rf", "svm"])
    if model_type == "svm":
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)    # параметр есть только у SVM
        ...
```

Основные методы:

| метод | что предлагает |
|---|---|
| `suggest_float(name, low, high, log=False)` | вещественное число; `log=True` — для величин «на порядки» |
| `suggest_int(name, low, high, step=1, log=False)` | целое |
| `suggest_categorical(name, choices)` | один из вариантов |

**Pruning (отсечение).** Если обучение итеративное (бустинг, нейросеть), objective может сообщать промежуточные значения через `trial.report(value, step)` и спрашивать `trial.should_prune()`. Прунер (например, `MedianPruner`) прерывает пробу, если на том же шаге она хуже медианы предыдущих. Плохие настройки отсеиваются за долю времени — в итоге при том же бюджете проверяется в несколько раз больше вариантов.

## Пример на Python

В среде курса Optuna не установлена (`pip install optuna`), код написан по её актуальному API:

```python
import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=0)

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10, log=True),
    }
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0, **params)
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=50)

print(study.best_params)
print(round(study.best_value, 4))
df = study.trials_dataframe()                       # вся история проб
# optuna.visualization.plot_param_importances(study)  — какие параметры важнее
```

Тот же поиск в Hyperopt выглядит так — пространство задаётся отдельно, а функция **минимизируется**:

```python
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.3)),
    "max_leaf_nodes": hp.qloguniform("max_leaf_nodes", np.log(8), np.log(128), 1),
}

def hyperopt_objective(p):
    model = HistGradientBoostingClassifier(learning_rate=p["learning_rate"],
                                           max_leaf_nodes=int(p["max_leaf_nodes"]))
    auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
    return {"loss": -auc, "status": STATUS_OK}       # минимизируем −AUC

best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=50, trials=Trials())
```

## Полезные возможности Optuna

- **Хранилище**: `create_study(storage="sqlite:///study.db", study_name="exp1", load_if_exists=True)` — можно прервать и продолжить, а несколько процессов могут работать над одним study параллельно.
- **Визуализация**: история оптимизации, важность параметров, срезы по каждому параметру.
- **Многокритериальная оптимизация**: `directions=["maximize", "minimize"]` — например, качество и время предсказания; результат — фронт Парето `study.best_trials`.
- **Интеграции** с LightGBM, XGBoost, CatBoost, PyTorch Lightning — колбэки для pruning.
- Другие сэмплеры: `RandomSampler`, `CmaEsSampler` (для непрерывных пространств), `GPSampler`.

## Подводные камни

- **Мало проб.** Первые ~10 проб TPE делает случайно (`n_startup_trials`). При бюджете 15–20 проб он почти не отличается от Random Search — выигрыш появляется на десятках и сотнях.
- **Утечка и переобучение на CV.** Сотни проб на одних и тех же фолдах — тоже подгонка: лучший `best_value` оптимистичен. Финальную оценку делайте на отложенном тесте.
- **Слишком широкие или неверные шкалы.** Learning rate без `log=True` почти всегда тратит большую часть проб на крупные значения.
- **Недетерминированность.** Без `seed` у сэмплера и `random_state` у модели результаты не повторяются.
- **Pruning только при честных промежуточных значениях.** Если сообщать метрику на обучающей выборке, прунер будет отсекать не те пробы.

## Итоги

- Optuna и Hyperopt реализуют байесовскую оптимизацию с алгоритмом TPE: моделируются плотности хороших и плохих проб.
- В Optuna пространство поиска задаётся кодом внутри objective (`suggest_float`, `suggest_int`, `suggest_categorical`), поэтому легко делать условные параметры.
- Pruning досрочно останавливает плохие пробы и экономит основную часть времени.
- Хранилище в базе данных даёт продолжение, параллельность и историю экспериментов.
- Выигрыш над случайным поиском проявляется при десятках и сотнях проб; финальную оценку делайте на отложенном тесте.

## Проверьте себя

1. Как TPE решает, какую точку попробовать следующей?
2. Что такое pruning и почему он позволяет проверить больше вариантов за то же время?
3. Как в Optuna описать параметр, который имеет смысл только при определённом значении другого параметра?
