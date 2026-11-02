#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений по подготовке данных: кодирование категорий, пропуски, PCA,
несбалансированные классы, выбросы и цикл «Отбор признаков» (feature_selection/exercise1…7).

Входные данные этих упражнений исторические (не синтетические с seed) — они сохраняются
байт в байт, чтобы прежние решения учащихся оставались верными. Генератор переписывает
task.md (условие + «Проверка» + «Разбор»), solution.py и answer.key.

Проверка решения, как и у остальных упражнений:
    python tools/check_exercise.py exercises/<имя>

Запуск из корня репозитория:
    python tools/generate_data_exercises.py
"""

import io
import textwrap

import pandas as pd

from generate_basic_exercises import EXERCISES, build

# ---------------------------------------------------------------- данные


def from_text(text):
    """Небольшая таблица, встроенная в генератор. Всё читается как строки — CSV пишется без изменений."""
    return pd.read_csv(io.StringIO(textwrap.dedent(text).strip() + "\n"), dtype=str, keep_default_na=False)


def existing(name):
    """Текущий input.csv упражнения (большие исторические данные) — без изменений, как строки."""
    return lambda rng: pd.read_csv(EXERCISES / name / "input.csv", dtype=str, keep_default_na=False)


def data_categorical(rng):
    return from_text("""
        product_id,category,size,color,price
        1,Electronics,M,Red,100
        2,Clothing,L,Blue,50
        3,Electronics,S,Red,80
        4,Furniture,L,Green,200
        5,Clothing,M,Blue,45
        6,Electronics,L,Red,120
        7,Furniture,S,Green,150
        8,Clothing,M,Red,55
    """)


def data_missing(rng):
    return from_text("""
        name,age,salary,city,experience
        Alice,25,50000,Moscow,3
        Bob,,60000,SPB,5
        Charlie,30,,Moscow,
        David,28,55000,,4
        Eve,35,70000,Moscow,7
        Frank,,65000,SPB,
        George,32,58000,Moscow,6
    """)


def data_pca(rng):
    return from_text("""
        id,feature1,feature2,feature3,feature4
        1,2.5,2.4,3.0,2.8
        2,0.5,0.7,1.0,0.8
        3,2.2,2.9,2.5,2.6
        4,1.9,2.2,2.0,2.1
        5,3.1,3.0,3.5,3.2
        6,2.3,2.7,2.8,2.5
        7,2.0,1.6,2.2,2.0
        8,1.0,1.1,1.5,1.2
        9,1.5,1.6,1.8,1.7
        10,1.1,0.9,1.3,1.0
    """)


def data_imbalanced(rng):
    return from_text("""
        id,feature1,feature2,feature3,class
        1,1.2,3.4,2.1,0
        2,2.3,4.5,3.2,0
        3,1.8,3.9,2.8,0
        4,2.1,4.2,3.0,0
        5,1.5,3.6,2.5,0
        6,2.5,4.8,3.5,0
        7,1.9,4.0,2.9,0
        8,2.2,4.3,3.1,0
        9,5.0,8.0,6.0,1
        10,4.8,7.8,5.8,1
    """)


def data_outliers(rng):
    return from_text("""
        id,feature1,feature2,feature3
        1,10,20,30
        2,15,25,35
        3,12,22,32
        4,100,23,31
        5,14,200,33
        6,13,21,300
        7,11,24,34
        8,16,26,36
        9,14,23,32
        10,15,25,35
    """)


def data_corr_target(rng):
    return from_text("""
        feature1,feature2,feature3,feature4,feature5,target
        5,10,2,8,3,1
        7,15,3,9,4,1
        3,8,1,7,2,0
        9,20,4,10,5,1
        2,6,1,6,1,0
        8,18,3,9,4,1
        4,9,2,7,3,0
        6,14,2,8,3,1
        1,5,0,5,1,0
        10,22,5,11,6,1
    """)


FS = "feature_selection/"
FS_FORMAT = "CSV с одной колонкой `feature` — названия признаков (например, `feature2`), по одному в строке."

# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "categorical_encoding",
        "title": "Кодирование категориальных признаков",
        "data": data_categorical,
        "task": """
            Дан файл `input.csv` с данными о 8 товарах: `product_id`, `category` (Clothing / Electronics / Furniture),
            `size` (S / M / L), `color` (Red / Blue / Green) и `price`.

            Закодируйте категориальные признаки тремя разными способами:

            1. **Порядковое (ordinal) кодирование** `size` → `size_encoded`: S = 0, M = 1, L = 2
               (по возрастанию размера — это **не** алфавитный порядок).
            2. **One-hot кодирование** `category` → три колонки `category_Clothing`, `category_Electronics`,
               `category_Furniture`: 1, если товар из этой категории, иначе 0.
            3. **Target encoding** `color` → `color_target_encoded`: каждый цвет заменяется средней ценой `price`
               товаров этого цвета (среднее по всем 8 строкам файла), округлённой до 2 знаков.
            4. Колонку `price` оставьте без изменений; `product_id` и исходные `category`, `size`, `color` в ответ не входят.

            ## Подсказки

            - У размеров есть естественный порядок, а у категорий — нет. Поэтому `size` можно заменить числами,
              а для `category` числа 0/1/2 внесли бы ложный порядок — нужны отдельные бинарные колонки.
            - `LabelEncoder` нумерует значения по алфавиту (L = 0, M = 1, S = 2) — здесь нужен свой словарь
              и `.map()`. Для one-hot удобен `pd.get_dummies(..., prefix='category', dtype=int)`.
            - Среднее по группе, «растянутое» обратно на каждую строку, даёт
              `df.groupby('color')['price'].transform('mean')`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `size_encoded, category_Clothing, category_Electronics, category_Furniture,
            color_target_encoded, price` (именно в этом порядке) — 8 строк в порядке `input.csv`.
            `size_encoded` и one-hot колонки — целые числа, `color_target_encoded` округлить до 2 знаков.
        """,
        "solution": """
            from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

            SIZE_ORDER = ['S', 'M', 'L']   # порядок по смыслу, а не по алфавиту


            def main():
                df = pd.read_csv('input.csv')
                result = pd.DataFrame()

                # 1. Порядковое кодирование: явный словарь S→0, M→1, L→2
                result['size_encoded'] = df['size'].map({s: i for i, s in enumerate(SIZE_ORDER)})
                # Сверка: OrdinalEncoder с заданным порядком категорий
                ordinal = OrdinalEncoder(categories=[SIZE_ORDER]).fit_transform(df[['size']])[:, 0]
                assert (ordinal == result['size_encoded']).all()

                # 2. One-hot: по одной бинарной колонке на категорию (get_dummies сортирует по алфавиту)
                dummies = pd.get_dummies(df['category'], prefix='category', dtype=int)
                result = pd.concat([result, dummies], axis=1)
                onehot = OneHotEncoder(sparse_output=False).fit_transform(df[['category']])
                assert (onehot == dummies.to_numpy()).all()

                # 3. Target encoding: средняя цена своего цвета в каждой строке
                means = df.groupby('color')['price'].mean()
                print("Средняя цена по цвету:", means.round(2).to_dict())
                result['color_target_encoded'] = df['color'].map(means).round(2)
                assert (result['color_target_encoded']
                        == df.groupby('color')['price'].transform('mean').round(2)).all()

                result['price'] = df['price']
                return result
        """,
        "review": """
            **Идея.** Способ кодирования зависит от природы признака. Порядковый признак (S < M < L) кодируется
            числами по порядку; номинальный (категория товара) — one-hot, чтобы модель не увидела несуществующий
            порядок; target encoding сжимает категорию в одно число — среднее целевой переменной по ней.

            **Типичные ошибки:**
            - `LabelEncoder` для `size` — получится L = 0, M = 1, S = 2: порядок перевёрнут и перемешан, `size_encoded` не совпадёт.
            - `get_dummies(..., drop_first=True)` — пропадёт колонка `category_Clothing` (так делают для линейных
              моделей против мультиколлинеарности, но здесь нужны все три).
            - Среднее цены по всей таблице или медиана вместо среднего по цвету — у всех цветов одинаковые/другие числа.

            **Что вынести:** target encoding использует целевую переменную, поэтому на реальной задаче его считают
            только по train (а лучше out-of-fold), иначе в признак утекает ответ. → шпаргалка «Кодирование категориальных признаков».
        """,
    },
    {
        "name": "data_preprocessing_missing_encoding",
        "title": "Обработка пропущенных значений и кодирование",
        "data": data_missing,
        "task": """
            Дан файл `input.csv` с данными о 7 сотрудниках: `name`, `age`, `salary`, `city` (Moscow / SPB),
            `experience`. В `age`, `salary`, `city` и `experience` есть пропуски (пустые ячейки).

            1. Удалите колонку `name` — для анализа она не нужна.
            2. Заполните пропуски:
               - в `age`, `salary`, `experience` — **средним** значением по известным (непустым) значениям этой колонки;
               - в `city` — **модой** (самым частым городом).
            3. Примените one-hot кодирование к `city`: колонки `city_Moscow` и `city_SPB` (1, если город совпадает,
               иначе 0); исходную колонку `city` удалите.

            ## Подсказки

            - `df.isna().sum()` покажет, сколько пропусков в каждой колонке. `mean()` в pandas пропуски
              игнорирует — среднее и так считается только по известным значениям.
            - Мода: `df['city'].mode()[0]` — `mode()` возвращает Series, потому что самых частых значений может быть несколько.
            - Для каждой числовой колонки — `df[col].fillna(df[col].mean())`, для города — `fillna` модой,
              затем `pd.get_dummies(df, columns=['city'], dtype=int)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `age, salary, experience, city_Moscow, city_SPB` — 7 строк в порядке `input.csv`.
            `age` и `experience` — целые числа (среднее округлить до целого), `salary` округлить до 2 знаков,
            `city_*` — 0 или 1.
        """,
        "solution": """
            from sklearn.impute import SimpleImputer


            def main():
                df = pd.read_csv('input.csv').drop(columns='name')
                print("Пропуски по колонкам:", df.isna().sum().to_dict())

                # Числовые колонки: среднее по известным значениям (pandas пропускает NaN сам)
                means = {col: df[col].mean() for col in ['age', 'salary', 'experience']}
                print("Средние:", {k: round(float(v), 2) for k, v in means.items()})
                # Категориальная колонка: мода
                mode_city = df['city'].mode()[0]
                print("Мода city:", mode_city)

                filled = df.fillna({**means, 'city': mode_city})

                # Сверка с SimpleImputer: та же логика «из коробки»
                num = SimpleImputer(strategy='mean').fit_transform(df[list(means)])
                assert np.allclose(num, filled[list(means)])
                cat = SimpleImputer(strategy='most_frequent').fit_transform(df[['city']])[:, 0]
                assert (cat == filled['city']).all()

                filled['age'] = filled['age'].round().astype(int)
                filled['experience'] = filled['experience'].round().astype(int)
                filled['salary'] = filled['salary'].round(2)

                # One-hot для города (колонки по алфавиту: Moscow, SPB)
                result = pd.get_dummies(filled, columns=['city'], dtype=int)
                return result[['age', 'salary', 'experience', 'city_Moscow', 'city_SPB']]
        """,
        "review": """
            **Идея.** Пропуск заполняют «типичным» значением колонки: для чисел — средним (или медианой, если есть
            выбросы), для категорий — модой. После заполнения категорию можно кодировать one-hot без колонки «NaN».

            **Типичные ошибки:**
            - `fillna(0)` для чисел — возраст 0 и зарплата 0 вместо средних 30 и 59666.67.
            - Медиана вместо среднего — `salary` станет 59000, а не 59666.67.
            - Пропуск в `city` как отдельная категория (`dummy_na=True` или «Unknown») — лишняя колонка, а у David
              обе `city_*` равны 0.
            - Среднее, посчитанное **после** заполнения нулями, — оно занижено.

            **Что вынести:** в реальной задаче среднее/моду считают только по train (`SimpleImputer.fit` на train,
            `transform` на test) — иначе утечка данных. → шпаргалка «Предобработка данных (пропуски и кодирование)».
        """,
    },
    {
        "name": "dimensionality_reduction",
        "title": "Снижение размерности: PCA",
        "data": data_pca,
        "task": """
            Дан файл `input.csv`: `id` и четыре признака `feature1 … feature4` у 10 объектов.

            1. **Стандартизируйте** признаки: z = (x − μ) / σ, где σ — стандартное отклонение **без поправки**
               (ddof = 0, как в `StandardScaler`).
            2. Примените **PCA** и оставьте 2 главные компоненты: собственные векторы ковариационной матрицы
               стандартизованных данных с двумя наибольшими собственными значениями.
            3. Спроецируйте стандартизованные данные на эти векторы: PC1, PC2.

            **Знак компонент.** Собственный вектор определён с точностью до знака (v и −v одинаково хороши).
            Выберите знак так, чтобы **наибольшая по модулю координата** каждого вектора нагрузок была
            **положительной** — так делает `sklearn.decomposition.PCA` (версии 1.5+).

            Можно решить вручную через NumPy, можно через `StandardScaler` + `PCA(n_components=2)` — ответ один.

            ## Подсказки

            - Посмотрите на корреляции признаков: все четыре почти линейно связаны. Как вы думаете, какую долю
              дисперсии возьмёт на себя PC1?
            - `np.linalg.eigh(np.cov(Z, rowvar=False))` возвращает собственные значения **по возрастанию** —
              порядок нужно перевернуть, чтобы PC1 была первой.
            - Возьмите матрицу W (4×2) из двух главных собственных векторов, приведите знак каждого столбца
              (если его наибольший по модулю элемент отрицателен — умножьте столбец на −1) и посчитайте `Z @ W`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, PC1, PC2` — 10 строк в порядке `input.csv`; PC1 и PC2 округлить до 4 знаков.
        """,
        "solution": """
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['feature1', 'feature2', 'feature3', 'feature4']


            def main():
                df = pd.read_csv('input.csv')
                X = df[FEATURES].to_numpy()

                # 1. Стандартизация (ddof=0 — как в StandardScaler)
                Z = (X - X.mean(axis=0)) / X.std(axis=0)

                # 2. Собственные векторы ковариационной матрицы; eigh сортирует значения по возрастанию
                eigvals, eigvecs = np.linalg.eigh(np.cov(Z, rowvar=False))
                order = np.argsort(eigvals)[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                print("Доля объяснённой дисперсии:", (eigvals / eigvals.sum()).round(4))

                W = eigvecs[:, :2]
                # Знак: наибольшая по модулю координата каждого вектора — положительная
                for k in range(W.shape[1]):
                    if W[np.argmax(np.abs(W[:, k])), k] < 0:
                        W[:, k] *= -1
                print("Нагрузки PC1:", W[:, 0].round(3), " PC2:", W[:, 1].round(3))

                # 3. Проекция
                P = Z @ W

                # Сверка с sklearn (модули — на случай старых версий с другим правилом знака)
                sk = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
                assert np.allclose(np.abs(P), np.abs(sk))

                return pd.DataFrame({'id': df['id'], 'PC1': P[:, 0].round(4), 'PC2': P[:, 1].round(4)})
        """,
        "review": """
            **Идея.** PCA поворачивает систему координат так, чтобы первая ось шла вдоль наибольшего разброса данных,
            вторая — вдоль наибольшего из оставшихся и т.д. Здесь признаки почти дублируют друг друга, поэтому PC1
            (нагрузки ≈ 0.5 у всех четырёх — по сути среднее признаков) объясняет ~97% дисперсии.

            **Типичные ошибки:**
            - Взять последние столбцы `eigh` без разворота порядка — получатся компоненты с **наименьшей** дисперсией.
            - Стандартизация с ddof = 1 (`df.std()`) — все проекции умножатся на √(9/10) ≈ 0.949.
            - Не привести знак — у PC1 или PC2 все значения с противоположным знаком.
            - PCA без стандартизации — другие нагрузки и числа (признак с большим разбросом «перетягивает» PC1).

            **Что вынести:** перед PCA признаки масштабируют; знак компоненты произволен, а смысл несут нагрузки
            и доля объяснённой дисперсии. → шпаргалка «Понижение размерности».
        """,
    },
    {
        "name": "imbalanced_data",
        "title": "Балансировка классов: random oversampling",
        "data": data_imbalanced,
        "task": """
            Дан файл `input.csv` для бинарной классификации: `id`, `feature1 … feature3` и `class`.
            Классы несбалансированы: объектов класса 0 гораздо больше, чем класса 1.

            1. Посчитайте число объектов каждого класса.
            2. Сделайте **random oversampling** класса 1: добавьте n₀ − n₁ копий строк класса 1, выбранных
               **случайно с возвращением**, чтобы классов стало поровну. Выбор строго такой:

               ```python
               np.random.seed(42)
               picked = np.random.choice(minority_ids, size=n_extra, replace=True)
               ```

               где `minority_ids` — массив `id` строк класса 1 в порядке файла, `n_extra` = n₀ − n₁.
            3. Копии получают **новые** `id`: max(id) + 1, max(id) + 2, … — в порядке, в котором строки выбраны
               в `picked`; признаки и `class` копируются из исходной строки.
            4. Ответ — все исходные строки плюс копии.

            ## Подсказки

            - Исходные строки остаются как есть; добавить нужно ровно столько копий, чтобы классы сравнялись.
            - Ответ зависит от конкретного генератора случайных чисел:`np.random.default_rng(42).choice(...)` даёт **другую**
              последовательность, чем `np.random.seed(42)` + `np.random.choice`. А вот
              `sklearn.utils.resample(..., replace=True, n_samples=n_extra, random_state=42)` совпадает с ним.
            - `df.set_index('id').loc[picked]` достаёт строки в порядке выборки (с повторами); затем `reset_index`,
              новые `id` и `pd.concat` с исходной таблицей.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, feature1, feature2, feature3, class` — исходные строки и копии,
            отсортированные по `id` по возрастанию. Значения признаков — как в `input.csv`.
        """,
        "solution": """
            from sklearn.utils import resample


            def main():
                df = pd.read_csv('input.csv')
                counts = df['class'].value_counts()
                print("Объектов по классам:", counts.to_dict())

                n_extra = counts[0] - counts[1]
                minority_ids = df.loc[df['class'] == 1, 'id'].to_numpy()

                # Случайный выбор с возвращением (воспроизводимо благодаря seed)
                np.random.seed(42)
                picked = np.random.choice(minority_ids, size=n_extra, replace=True)
                print("Выбраны id:", picked)
                # sklearn.utils.resample использует тот же генератор (RandomState) — результат совпадает
                assert (picked == resample(minority_ids, replace=True, n_samples=n_extra, random_state=42)).all()

                # Копии строк в порядке выборки, с новыми id
                copies = df.set_index('id').loc[picked].reset_index()
                copies['id'] = np.arange(df['id'].max() + 1, df['id'].max() + 1 + n_extra)

                result = pd.concat([df, copies], ignore_index=True).sort_values('id').reset_index(drop=True)
                print("После балансировки:", result['class'].value_counts().to_dict())
                return result
        """,
        "review": """
            **Идея.** Random oversampling просто дублирует объекты меньшего класса — модель видит класс 1 чаще,
            но новой информации не получает. Это эквивалентно увеличению веса класса (`class_weight`) и может вести
            к переобучению на конкретных точках; SMOTE вместо копий создаёт новые точки между соседями.

            **Типичные ошибки:**
            - `np.random.default_rng(42)` вместо `np.random.seed(42)` — выбираются другие строки, признаки копий не совпадут.
            - Выбор из всей таблицы, а не только из класса 1 — среди копий окажутся объекты класса 0.
            - Добавить n₀ копий вместо n₀ − n₁ — 18 строк вместо 16, класс 1 станет больше класса 0.
            - Выбор без возвращения — из 2 строк нельзя выбрать 6 разных, `choice` выдаст ошибку.

            **Что вынести:** балансировку делают только на обучающей выборке **после** разбиения — иначе копии
            одной точки попадут и в train, и в test. → шпаргалка «Imbalanced Data».
        """,
    },
    {
        "name": "outliers_handling",
        "title": "Обработка выбросов: метод IQR",
        "data": data_outliers,
        "task": """
            Дан файл `input.csv`: `id` и три признака `feature1, feature2, feature3` у 10 объектов; есть выбросы.

            1. Для **каждого** признака отдельно найдите границы по методу **IQR**:
               - Q1 и Q3 — 25-й и 75-й перцентили, посчитанные **с линейной интерполяцией** (по умолчанию в
                 `pandas.Series.quantile` и `np.percentile`);
               - IQR = Q3 − Q1; нижняя граница = Q1 − 1.5 · IQR, верхняя = Q3 + 1.5 · IQR.
               Границы считаются **один раз** по всем 10 строкам.
            2. Выброс — значение **строго** меньше нижней или **строго** больше верхней границы
               (значение, равное границе, выбросом не считается).
            3. Удалите строки, в которых есть выброс хотя бы в одном признаке.

            ## Подсказки

            - Посмотрите на данные глазами: в каждом признаке есть одно значение, на порядок больше остальных.
            - Квартили можно считать разными способами (линейная интерполяция, медианы половин, ближайшее значение) —
              Q1 и Q3 получатся немного разными. Нужен именно `df[col].quantile(0.25)` / `quantile(0.75)`.
            - Заведите булеву маску из одних True и для каждого признака сужайте её:
              `mask &= df[col].between(low, high)`; в ответ пойдут строки, где маска осталась True.

            ## Ожидаемый формат выходного файла

            CSV с одной колонкой `id` — id оставшихся строк по возрастанию.
        """,
        "solution": """
            FEATURES = ['feature1', 'feature2', 'feature3']


            def main():
                df = pd.read_csv('input.csv')
                keep = pd.Series(True, index=df.index)

                for col in FEATURES:
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    assert np.isclose(q1, np.percentile(df[col], 25)) and np.isclose(q3, np.percentile(df[col], 75))
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    inside = df[col].between(low, high)          # границы включительно
                    print(f"{col}: Q1={q1}, Q3={q3}, границы [{low}, {high}], выбросы id={df.loc[~inside, 'id'].tolist()}")
                    keep &= inside

                # Для сравнения: правило «3 сигмы» здесь не срабатывает — выброс сам раздувает σ
                f1 = df['feature1']
                print(f"feature1: mean + 3·std = {f1.mean() + 3 * f1.std(ddof=0):.1f} — значение 100 под порогом")

                return df.loc[keep, ['id']].sort_values('id').reset_index(drop=True)
        """,
        "review": """
            **Идея.** Квартили почти не зависят от экстремальных значений, поэтому границы Q1 − 1.5·IQR и
            Q3 + 1.5·IQR устойчивы к самим выбросам. Правило «среднее ± 3σ» на 10 точках не поймает даже 100
            в `feature1`: выброс так раздувает σ, что сам оказывается внутри границ.

            **Типичные ошибки:**
            - Объединить условия через «или» вместо «и» (`|` вместо `&`) — останутся строки с выбросом в одном из признаков.
            - Пересчитывать границы после каждого удаления — это другой, итеративный метод; на других данных он удалит больше.
            - Вывести id удалённых строк (4, 5, 6) вместо оставшихся.
            - Квартили «медианами половин» — другие Q1/Q3, на пограничных значениях другой ответ.

            **Что вынести:** IQR — простой и устойчивый фильтр для одного признака; но удалять выбросы стоит только
            если это ошибки данных, а не редкие настоящие случаи. → шпаргалка «Обработка выбросов».
        """,
    },
    # ------------------------------------------------------------ цикл «Отбор признаков»
    {
        "name": FS + "exercise1_variance_threshold",
        "title": "Отбор признаков 1: удаление низкодисперсных признаков (Variance Threshold)",
        "data": existing(FS + "exercise1_variance_threshold"),
        "task": f"""
            Дан файл `input.csv` с признаками `feature1 … feature6` (20 объектов, целевой переменной нет).

            1. Вычислите **дисперсию** каждого признака по формуле σ² = Σ(x − x̄)² / n
               (деление на n, т.е. ddof = 0 — как в `sklearn.feature_selection.VarianceThreshold`).
            2. Оставьте признаки с дисперсией **≥ 0.5**, остальные удалите.

            ## Подсказки

            - Сравните `df.describe()`: у части признаков значения почти не меняются от строки к строке.
            - `df.var()` в pandas по умолчанию делит на n − 1 (ddof = 1); формула в условии — `df.var(ddof=0)`.
            - `VarianceThreshold(threshold=0.5).fit(df).get_support()` даёт булеву маску оставленных колонок —
              сверьтесь с ручным расчётом.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} Оставшиеся признаки — в порядке их колонок в `input.csv` (feature1, feature2, …).
        """,
        "solution": """
            from sklearn.feature_selection import VarianceThreshold

            THRESHOLD = 0.5


            def main():
                df = pd.read_csv('input.csv')

                # Дисперсия с делением на n (ddof=0)
                variances = ((df - df.mean()) ** 2).mean()
                assert np.allclose(variances, df.var(ddof=0))
                print("Дисперсии:", variances.round(4).to_dict())

                kept = variances.index[variances >= THRESHOLD]

                # Сверка: VarianceThreshold оставляет признаки с дисперсией > threshold
                # (здесь ни одна дисперсия не равна 0.5, так что ≥ и > совпадают)
                selector = VarianceThreshold(threshold=THRESHOLD).fit(df)
                assert list(df.columns[selector.get_support()]) == list(kept)

                return pd.DataFrame({'feature': kept})
        """,
        "review": """
            **Идея.** Почти постоянный признак не может различать объекты, поэтому его можно выбросить, даже не глядя
            на целевую переменную. Это самый дешёвый фильтр: дисперсии feature2, feature4, feature6 — около 0.01.

            **Типичные ошибки:**
            - Стандартизировать данные **до** фильтра — после `StandardScaler` все дисперсии равны 1, и останутся все 6 признаков.
            - Вывести удалённые признаки вместо оставшихся.
            - Сравнивать со стандартным отклонением как с дисперсией без осознания разницы — для порогов около 1 ответ может поменяться.

            **Что вынести:** порог дисперсии зависит от масштаба признака, поэтому VarianceThreshold применяют к
            исходным (или одинаково масштабированным) данным, а для бинарных признаков — в долях p(1 − p).
            → шпаргалка «Отбор признаков».
        """,
    },
    {
        "name": FS + "exercise2_mutual_information",
        "title": "Отбор признаков 2: взаимная информация (Mutual Information)",
        "data": existing(FS + "exercise2_mutual_information"),
        "task": f"""
            Дан файл `input.csv`: признаки `feature1 … feature5` и бинарная целевая переменная `target` (100 объектов).

            1. Вычислите **взаимную информацию** (MI) каждого признака с `target` функцией
               `sklearn.feature_selection.mutual_info_classif(X, y, random_state=42)` — остальные параметры
               по умолчанию (все признаки непрерывные, `n_neighbors=3`).
            2. Выберите **3 признака** с наибольшей MI.

            MI ≥ 0 и равна 0 только для независимых величин; в отличие от корреляции, она замечает и нелинейные связи.

            ## Подсказки

            - Посчитайте для сравнения корреляцию Пирсона признаков с `target`: совпадает ли тройка лидеров? А порядок?
            - Для непрерывных признаков MI оценивается по k ближайшим соседям, а к данным добавляется крошечный
              шум — без `random_state` числа «плавают», и близкие признаки могут поменяться местами.
            - `pd.Series(mi, index=X.columns).nlargest(3)` — сразу три лидера в порядке убывания.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} 3 строки — в порядке **убывания** MI.
        """,
        "solution": """
            from functools import partial

            from sklearn.feature_selection import SelectKBest, mutual_info_classif


            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']

                mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
                print("MI:", mi.round(4).to_dict())
                # Для сравнения — линейная связь
                print("|r| Пирсона:", X.corrwith(y).abs().round(3).to_dict())

                top = mi.nlargest(3)

                # Сверка: SelectKBest с той же функцией выбирает тот же набор
                selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=3).fit(X, y)
                assert set(X.columns[selector.get_support()]) == set(top.index)

                return pd.DataFrame({'feature': top.index})
        """,
        "review": """
            **Идея.** MI измеряет, насколько знание признака уменьшает неопределённость в `target`. Это фильтр:
            признаки оцениваются по одному, без модели. Здесь тройка та же, что по корреляции (feature1, feature3,
            feature5), но порядок другой — MI видит у feature5 связь, которую линейная корреляция недооценивает.

            **Типичные ошибки:**
            - Без `random_state=42` — оценка MI слегка случайна, порядок близких признаков может поменяться.
            - `mutual_info_regression` для бинарного `target` — другая оценка и другие числа.
            - Оставить `target` в X — он займёт первое место с максимальной MI сам с собой.
            - Записать признаки по алфавиту, а не по убыванию MI.

            **Что вынести:** MI ловит любые зависимости, но оценивается шумно на малых выборках и не учитывает
            избыточность между признаками. → шпаргалка «Отбор признаков».
        """,
    },
    {
        "name": FS + "exercise3_rfe",
        "title": "Отбор признаков 3: рекурсивное удаление признаков (RFE)",
        "data": existing(FS + "exercise3_rfe"),
        "task": f"""
            Дан файл `input.csv`: признаки `feature1 … feature6` и бинарная целевая переменная `target` (150 объектов).

            Отберите **3 признака** методом **RFE** (Recursive Feature Elimination):

            ```python
            RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                n_features_to_select=3, step=1)
            ```

            обученным на всех 150 объектах. RFE обучает модель, удаляет признак с наименьшей важностью
            (`feature_importances_`), переобучает модель на оставшихся и повторяет, пока не останется 3 признака.

            ## Подсказки

            - Это wrapper-метод: после каждого удаления модель обучается заново, и важности пересчитываются —
              поэтому ответ может отличаться от «топ-3 по одному обучению».
            - При 6 признаках и `step=1` будет 3 удаления. `rfe.ranking_` покажет порядок: 1 — выбранные,
              большие числа — удалённые раньше.
            - `X.columns[rfe.support_]` — выбранные признаки. Можно написать цикл и вручную: обучить лес, убрать
              `argmin` важности, повторить — результат совпадёт с RFE.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} 3 строки — в порядке колонок в `input.csv` (feature1, feature2, …).
        """,
        "solution": """
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE


            def make_model():
                return RandomForestClassifier(n_estimators=100, random_state=42)


            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']

                # Вручную: обучаем лес, выкидываем самый неважный признак, повторяем
                features = list(X.columns)
                while len(features) > 3:
                    importances = make_model().fit(X[features], y).feature_importances_
                    weakest = features[int(np.argmin(importances))]
                    print(f"Удаляем {weakest}: важности {pd.Series(importances, index=features).round(3).to_dict()}")
                    features.remove(weakest)

                # Сверка с RFE
                rfe = RFE(make_model(), n_features_to_select=3, step=1).fit(X, y)
                print("ranking_:", pd.Series(rfe.ranking_, index=X.columns).to_dict())
                assert list(X.columns[rfe.support_]) == features

                return pd.DataFrame({'feature': features})
        """,
        "review": """
            **Идея.** RFE — wrapper-метод: качество признаков оценивает сама модель, и после каждого удаления
            важности пересчитываются (убрали один из коррелирующих признаков — важность второго вырастет).
            Здесь удаляются feature6, feature3, feature4, остаются feature1, feature2, feature5.

            **Типичные ошибки:**
            - Без `random_state=42` у леса — важности случайны, при близких значениях выбор меняется.
            - Взять топ-3 по одному обучению вместо рекурсии — здесь совпадает случайно, в общем случае нет.
            - Записать признаки в порядке важности или `ranking_`, а не в порядке колонок.

            **Что вынести:** RFE дороже фильтров (модель обучается много раз), зато учитывает взаимодействие
            признаков; число признаков лучше подбирать через `RFECV`. → шпаргалка «Отбор признаков».
        """,
    },
    {
        "name": FS + "exercise4_lasso",
        "title": "Отбор признаков 4: Lasso (L1-регуляризация)",
        "data": existing(FS + "exercise4_lasso"),
        "task": f"""
            Дан файл `input.csv`: признаки `feature1 … feature5` и непрерывная целевая переменная `target` (100 объектов).

            1. Стандартизируйте все признаки `StandardScaler` (`fit_transform` на всех 100 объектах; `target` не трогайте).
            2. Обучите `sklearn.linear_model.Lasso(alpha=0.1)` (остальные параметры по умолчанию) на стандартизованных данных.
            3. Отберите признаки с **ненулевыми** коэффициентами (`coef_ != 0`).

            ## Подсказки

            - Штраф Lasso — alpha · Σ|wᵢ|. Без стандартизации он по-разному давит на признаки разного масштаба.
            - Lasso обнуляет коэффициенты **точно** (в отличие от Ridge, который их только уменьшает) —
              достаточно сравнения `!= 0`, никакого своего порога не нужно.
            - `X.columns[lasso.coef_ != 0]`; попробуйте также alpha = 0.3 и 1 — как меняется набор?

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} Отобранные признаки — в порядке колонок в `input.csv` (feature1, feature2, …).
        """,
        "solution": """
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler


            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']
                Z = StandardScaler().fit_transform(X)

                lasso = Lasso(alpha=0.1).fit(Z, y)
                print("Коэффициенты:", pd.Series(lasso.coef_, index=X.columns).round(4).to_dict())
                selected = X.columns[lasso.coef_ != 0]

                # Чем больше alpha, тем больше нулей
                for alpha in [0.3, 1.0]:
                    coef = Lasso(alpha=alpha).fit(Z, y).coef_
                    print(f"alpha={alpha}: отобраны {list(X.columns[coef != 0])}")

                # Сверка: SelectFromModel с Lasso (для L1-моделей порог по умолчанию — почти ноль)
                sfm = SelectFromModel(Lasso(alpha=0.1)).fit(Z, y)
                assert list(X.columns[sfm.get_support()]) == list(selected)

                return pd.DataFrame({'feature': selected})
        """,
        "review": """
            **Идея.** L1-штраф заставляет оптимизатор обнулять коэффициенты слабых признаков — отбор встроен в обучение
            (embedded-метод). При alpha = 0.1 обнуляется только feature5; у feature2 коэффициент маленький (≈ 0.12),
            но ненулевой — при alpha = 0.3 исчезнет и он.

            **Типичные ошибки:**
            - Ridge вместо Lasso — коэффициенты не обнуляются, отобраны все 5 признаков.
            - Свой порог вроде `abs(coef) > 0.5` — пропадёт feature2, хотя его коэффициент не ноль.
            - Отбор по знаку (`coef > 0`) — потеряется feature4 с отрицательным коэффициентом.
            - Забыть стандартизацию — на этих данных признаки уже почти единичного масштаба и набор тот же,
              но на реальных данных выбор станет зависеть от единиц измерения.

            **Что вынести:** сила отбора задаётся alpha; её подбирают кросс-валидацией (`LassoCV`), а среди сильно
            коррелирующих признаков Lasso оставляет один почти случайно. → шпаргалка «Ridge, Lasso, ElasticNet».
        """,
    },
    {
        "name": FS + "exercise5_tree_importance",
        "title": "Отбор признаков 5: важность признаков в случайном лесе",
        "data": existing(FS + "exercise5_tree_importance"),
        "task": f"""
            Дан файл `input.csv`: признаки `feature1 … feature6` и бинарная целевая переменная `target` (200 объектов).

            1. Обучите `RandomForestClassifier(n_estimators=100, random_state=42)` (остальные параметры по умолчанию)
               на всех 200 объектах.
            2. Возьмите важности признаков `feature_importances_` (среднее уменьшение неопределённости Джини).
            3. Выберите **4 признака** с наибольшей важностью.

            ## Подсказки

            - Важность признака в дереве — суммарное уменьшение Джини во всех узлах, где по нему делали разбиение;
              в лесе она усредняется по деревьям и нормируется так, что сумма равна 1.
            - Без `random_state` бутстреп-выборки и выбор признаков в узлах случайны — у близких по важности
              признаков порядок может меняться.
            - `pd.Series(model.feature_importances_, index=X.columns).nlargest(4)`.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} 4 строки — в порядке **убывания** важности.
        """,
        "solution": """
            from sklearn.ensemble import RandomForestClassifier


            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']

                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                importances = pd.Series(model.feature_importances_, index=X.columns)
                print("Важности:", importances.round(4).to_dict())

                # Важность леса — среднее важностей отдельных деревьев (у каждого сумма = 1)
                per_tree = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
                assert np.allclose(per_tree, importances) and np.isclose(importances.sum(), 1)

                top = importances.nlargest(4)
                return pd.DataFrame({'feature': top.index})
        """,
        "review": """
            **Идея.** Лес сам «рассказывает», какие признаки чаще и полезнее использовались для разбиений. Это
            embedded-оценка: одна модель, никаких переобучений. Здесь feature4 и feature6 заметно слабее остальных
            (≈ 0.065 против 0.18–0.29).

            **Типичные ошибки:**
            - `RandomForestRegressor` вместо классификатора — другой критерий разбиения и другие важности.
            - Отсортировать по возрастанию или записать признаки в порядке колонок — порядок не совпадёт.
            - Без `random_state=42` — feature1 и feature2 (0.183 и 0.193) могут поменяться местами.

            **Что вынести:** MDI-важность завышает признаки с большим числом уникальных значений и считается на
            train; для честной оценки есть permutation importance на валидации. → шпаргалка «Отбор признаков».
        """,
    },
    {
        "name": FS + "exercise6_remove_correlated",
        "title": "Отбор признаков 6: удаление коррелирующих признаков",
        "data": existing(FS + "exercise6_remove_correlated"),
        "task": f"""
            Дан файл `input.csv` с признаками `feature1 … feature6` (100 объектов).

            1. Вычислите матрицу **корреляций Пирсона** между признаками.
            2. Рассмотрите каждую пару признаков (i, j), где i идёт в таблице **раньше** j (верхний треугольник
               матрицы без диагонали). Если |r(i, j)| > 0.8, удалите **более поздний** признак j.
               Признак удаляется, если он сильно коррелирует хотя бы с одним более ранним признаком
               (даже если тот ранний сам удалён).
            3. Оставьте остальные признаки.

            Пример: если |r(feature1, feature3)| = 0.85, удаляется feature3.

            ## Подсказки

            - `df.corr()` по умолчанию считает Пирсона; берите модуль — сильная отрицательная корреляция
              означает ту же избыточность.
            - Маска верхнего треугольника: `np.triu(np.ones(corr.shape, dtype=bool), k=1)` — каждая пара один раз,
              без диагонали (на диагонали всегда 1).
            - `upper = corr.abs().where(mask)`; удаляются колонки, где `(upper[col] > 0.8).any()`.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} Оставшиеся признаки — в порядке колонок в `input.csv` (feature1, feature2, …).
        """,
        "solution": """
            THRESHOLD = 0.8


            def main():
                df = pd.read_csv('input.csv')
                corr = df.corr().abs()
                cols = list(df.columns)

                # Вручную: двойной цикл по парам i < j
                to_drop = set()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        if corr.iloc[i, j] > THRESHOLD:
                            print(f"|r({cols[i]}, {cols[j]})| = {corr.iloc[i, j]:.3f} → удаляем {cols[j]}")
                            to_drop.add(cols[j])

                # То же через маску верхнего треугольника
                upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                assert to_drop == {c for c in upper.columns if (upper[c] > THRESHOLD).any()}

                kept = [c for c in cols if c not in to_drop]
                return pd.DataFrame({'feature': kept})
        """,
        "review": """
            **Идея.** Сильно коррелирующие признаки несут почти одну и ту же информацию; из каждой такой пары
            достаточно одного. Здесь три пары: (feature1, feature3), (feature2, feature4), (feature5, feature6) —
            удаляются вторые элементы.

            **Типичные ошибки:**
            - Не исключить диагональ — каждый признак коррелирует сам с собой на 1, удалятся все.
            - Полная матрица вместо верхнего треугольника — из каждой пары удаляются **оба** признака, ответ пустой.
            - Корреляция без модуля — сильные отрицательные связи будут пропущены (здесь все сильные связи
              положительные, но на других данных это ошибка).

            **Что вынести:** какой из пары оставить — решение по смыслу (дешевле, понятнее, меньше пропусков);
            порядок колонок — лишь простое формальное правило. → шпаргалка «Отбор признаков».
        """,
    },
    {
        "name": FS + "exercise7_correlation_target",
        "title": "Отбор признаков 7: корреляция с целевой переменной",
        "data": data_corr_target,
        "task": f"""
            Дан файл `input.csv`: признаки `feature1 … feature5` и бинарная целевая переменная `target` (10 объектов).

            1. Вычислите **корреляцию Пирсона** каждого признака с `target`:
               r = Σ(x − x̄)(y − ȳ) / √(Σ(x − x̄)² · Σ(y − ȳ)²).
            2. Выберите **3 признака** с наибольшим **модулем** |r|.

            ## Подсказки

            - Для бинарного `target` корреляция Пирсона тоже работает (это точечно-бисериальная корреляция).
            - `df.corr()['target']` — корреляции всех колонок с target, включая сам target (r = 1) — его нужно убрать.
            - Отсортируйте по `abs()` по убыванию и возьмите три первых. Не округляйте r до сортировки:
              у двух признаков |r| отличается лишь в третьем знаке.

            ## Ожидаемый формат выходного файла

            {FS_FORMAT} 3 строки — в порядке **убывания** |r|.
        """,
        "solution": """
            def main():
                df = pd.read_csv('input.csv')
                X, y = df.drop(columns='target'), df['target']

                # Вручную по формуле Пирсона
                dx, dy = X - X.mean(), y - y.mean()
                r = (dx.mul(dy, axis=0)).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
                assert np.allclose(r, df.corr()['target'].drop('target'))
                print("r с target:", r.round(4).to_dict())

                top = r.abs().sort_values(ascending=False).head(3)
                return pd.DataFrame({'feature': top.index})
        """,
        "review": """
            **Идея.** Самый простой фильтр для отбора: оценить линейную связь каждого признака с целевой переменной
            и взять сильнейшие. Знак r не важен для отбора — важна сила связи, поэтому сортируют по модулю.

            **Типичные ошибки:**
            - Оставить `target` в списке — он займёт первое место с r = 1.
            - Округлить r до 2 знаков перед сортировкой — feature4 (0.825) и feature2 (0.820) превратятся в
              одинаковые 0.82, и порядок станет случайным.
            - Сортировать по r без модуля — признаки с сильной отрицательной связью окажутся в конце (здесь все r > 0,
              но на других данных ответ изменится).

            **Что вынести:** корреляция с target видит только линейные связи и не замечает избыточности между
            самими признаками (здесь все пять попарно коррелируют сильнее 0.95 — по сути один признак) — комбинируйте её с
            удалением коррелирующих признаков. → шпаргалка «Отбор признаков».
        """,
    },
]


def main():
    print("Генерация упражнений по подготовке данных:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=401):
        build(exercise, seed)


if __name__ == "__main__":
    main()
