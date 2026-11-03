# Пайплайны ML

<!-- время: 15 мин -->

> **О чём лекция.** Как собрать предобработку и модель в один объект с помощью `Pipeline` и `ColumnTransformer` в scikit-learn, почему это защищает от утечки данных, как подбирать гиперпараметры шагов и чем такой пайплайн отличается от оркестрации вроде Airflow.

## Зачем это нужно

Реальная модель почти никогда не состоит из одного `fit`. Сначала нужно заполнить пропуски, закодировать категории, масштабировать числа, иногда сократить размерность — и только потом обучить модель. Если писать эти шаги вручную, легко допустить три ошибки:

- **утечка данных**: скейлер обучили на всей выборке, включая тест, — и оценка качества оказалась завышенной;
- **несогласованность**: на обучении шаги применили в одном порядке, а в сервисе — в другом или забыли один;
- **неудобство**: при кросс-валидации предобработку приходится повторять внутри каждого фолда руками.

`Pipeline` решает все три проблемы: цепочка шагов становится одним объектом с методами `fit` и `predict`.

## Интуиция: конвейер на заводе

Пайплайн похож на заводской конвейер. Сырьё (исходная таблица) проходит станки по очереди: один чистит, другой режет, третий красит, и в конце получается готовая деталь (предсказание). Важно, что **настраивают станки только на обучающем сырье**: скейлер запоминает средние и стандартные отклонения обучающей выборки, кодировщик — список известных категорий. На новых данных станки уже ничего не перенастраивают, а только применяют то, что запомнили.

Именно это и даёт `Pipeline`: вызов `fit` последовательно делает `fit_transform` на каждом шаге и `fit` на последнем, а вызов `predict` — только `transform` и `predict`.

## Как это работает

Пайплайн — это список пар «имя → шаг». Все шаги, кроме последнего, должны быть трансформерами (иметь `fit` и `transform`), последний может быть моделью:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)   # scale.fit_transform → model.fit
pipe.predict(X_test)         # scale.transform → model.predict
```

Если имена шагов не важны, есть сокращение `make_pipeline(StandardScaler(), LogisticRegression())`: имена создаются автоматически из названий классов в нижнем регистре.

## Разные столбцы — разная обработка

В табличных данных числовые и категориальные признаки обрабатываются по-разному. Для этого есть `ColumnTransformer`: он применяет свой трансформер к своему набору столбцов и склеивает результаты.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
prep = ColumnTransformer([
    ("num", num, ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
```

Пайплайны вкладываются друг в друга: здесь числовая ветка сама является пайплайном из двух шагов. А `FeatureUnion`, наоборот, применяет несколько трансформеров к **одним и тем же** данным параллельно и объединяет их выходы — например, главные компоненты плюс несколько отобранных признаков.

## Подбор гиперпараметров шагов

Гиперпараметр любого шага задаётся через двойное подчёркивание: `имя_шага__параметр`. Для вложенных шагов имена складываются в цепочку:

```python
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "prep__num__impute__strategy": ["mean", "median"],
}
grid = GridSearchCV(pipe, param_grid, cv=5)
```

Главное достоинство: `GridSearchCV` и `cross_val_score` на каждом фолде заново обучают **весь** пайплайн. Импьютер и скейлер видят только обучающую часть фолда, а валидационная остаётся «новыми данными». Без пайплайна получить честную кросс-валидацию с предобработкой гораздо труднее.

## Собственный шаг

Если нужного трансформера нет, его легко написать: наследуйтесь от `BaseEstimator` и `TransformerMixin` и реализуйте `fit` (запоминает статистики и возвращает `self`) и `transform`:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(X)
```

`TransformerMixin` даёт метод `fit_transform` бесплатно, а `BaseEstimator` — `get_params`/`set_params`, без которых шаг не будет работать в `GridSearchCV`. Параметры конструктора нужно сохранять в атрибуты с теми же именами и не менять их в `__init__`.

## Пайплайн модели и пайплайн данных

Слово «пайплайн» в ML означает две разные вещи:

| | sklearn `Pipeline` | оркестрация (Airflow, Kubeflow, Prefect) |
|---|---|---|
| что это | один объект: предобработка + модель | граф задач: выгрузка, проверки, обучение, отчёты |
| где живёт | в памяти, сохраняется в файл | на сервере-планировщике |
| когда запускается | при `fit` и `predict` | по расписанию или событию |
| что решает | согласованность и отсутствие утечки | автоматизация, повторы при сбоях, зависимости |

Шаги оркестрируемого пайплайна стоит делать **идемпотентными**: повторный запуск с теми же входами даёт тот же результат и не дублирует данные. Тогда упавшую задачу можно просто перезапустить.

## Пример на Python

Полный пайплайн для таблицы с пропусками и категориями, подбор `C` и предсказание для объекта с неизвестным городом:

```python
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame({
    "age": rng.integers(18, 70, n).astype(float),
    "income": rng.lognormal(10, 0.5, n),
    "city": rng.choice(["Москва", "Казань", "Томск"], n),
})
df.loc[rng.random(n) < 0.1, "income"] = np.nan       # 10% пропусков
y = ((df["age"] > 45) | (df["city"] == "Томск")).astype(int)

num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
prep = ColumnTransformer([
    ("num", num, ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, random_state=0)
grid = GridSearchCV(pipe, {"model__C": [0.01, 0.1, 1, 10]}, cv=5).fit(X_tr, y_tr)
print(grid.best_params_, round(grid.score(X_te, y_te), 3))   # {'model__C': 10} 0.993

new = pd.DataFrame({"age": [55.0], "income": [np.nan], "city": ["Самара"]})
print(grid.predict(new))   # [1] — пропуск заполнен, незнакомый город не сломал модель
```

`grid.best_estimator_` — готовый обученный пайплайн; его и сохраняют через `joblib.dump`.

## Подводные камни

- **Предобработка до разбиения.** `StandardScaler().fit(X)` на всей выборке, а потом `train_test_split` — классическая утечка. Всё, что учится на данных, должно быть внутри пайплайна.
- **Неизвестные категории.** `OneHotEncoder` по умолчанию падает на категории, которой не было в обучении. В продакшене почти всегда нужен `handle_unknown="ignore"`.
- **Опечатка в имени параметра.** `model_C` вместо `model__C` вызывает ошибку; список доступных имён даёт `pipe.get_params().keys()`.
- **Кастомный шаг меняет параметры в `__init__`.** Тогда клонирование в `GridSearchCV` работает неправильно.
- **Отбор признаков вне пайплайна.** Если отбирать признаки по корреляции с целевой переменной на всей выборке, это тоже утечка; `SelectKBest` должен стоять внутри.

## Итоги

- `Pipeline` превращает цепочку предобработки и модель в один объект с `fit` и `predict`.
- Трансформеры учатся только на обучающих данных, поэтому пайплайн защищает от утечки, в том числе внутри кросс-валидации.
- `ColumnTransformer` обрабатывает разные столбцы по-разному, `FeatureUnion` объединяет несколько представлений.
- Гиперпараметры шагов задаются как `шаг__параметр`.
- Сохранять в продакшен нужно весь пайплайн; оркестрация (Airflow и др.) — это другой уровень, автоматизация задач вокруг него.

## Проверьте себя

1. Почему масштабирование до `train_test_split` завышает оценку качества и как пайплайн это предотвращает?
2. Как задать параметр `strategy` импьютера, который находится в шаге `num` внутри `ColumnTransformer` с именем `prep`?
3. Что нужно реализовать в собственном трансформере, чтобы он работал внутри `GridSearchCV`?
