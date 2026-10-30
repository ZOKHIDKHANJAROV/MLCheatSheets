# 💪 Упражнения по машинному обучению

> **Практические задания для закрепления теории из [cheatsheets](README.md)**

Этот файл содержит набор упражнений для практического освоения различных методов предобработки и работы с данными в машинном обучении.

## 🎯 О практических упражнениях

Упражнения разработаны для **отработки навыков** после изучения соответствующих cheatsheets. Каждое упражнение включает:

- 📁 **Реальные данные** - CSV файлы для практики
- 📝 **Подробное условие** - что нужно сделать
- ✅ **Правильные ответы** - для самопроверки
- 💡 **Готовые решения** - с комментариями и объяснениями (для некоторых упражнений)

### ✨ Новое: Подробные решения!

Для некоторых упражнений доступны файлы `solution.py` с подробными комментариями. Используйте их для:
- 📖 Изучения best practices
- 🔍 Понимания пошагового процесса решения
- ✔️ Проверки своего подхода
- 🎓 Обучения на примерах

### Связь с курсом

Упражнения соответствуют разделам из [Дорожной карты](ROADMAP.md):
- **Этап 3 (Путь 1: Полный новичок)**: Упражнения 1-8
- **Этапы 2, 4–6 (Путь 1)**: Упражнения 9-15 (метрики, регрессия, классификация, деревья, бустинг)
- **Этапы 1, 3, 5 (Путь 2)**: Упражнения 16-22 (активации, softmax, backprop, MLP, оптимизаторы, свёртка, attention)
- **Подготовка данных**: Все упражнения полезны
- **Feature Engineering**: Упражнения 6.1-6.7

## Список упражнений

### 1. Предобработка данных (Preprocessing)

**Описание:** Применение StandardScaler к числовым признакам и LabelEncoder к категориальным признакам.

**Файлы:**
- [Входные данные](exercises/preprocessing/input.csv)
- [Условие задачи](exercises/preprocessing/task.md)
- [Ответ](exercises/preprocessing/answer.csv)
- [Решение с комментариями](exercises/preprocessing/solution.py) ✨ NEW!

---

### 2. Обработка пропущенных значений и кодирование (Data Preprocessing: Missing Values & Encoding)

**Описание:** Заполнение пропущенных значений (mean для числовых, mode для категориальных) и применение One-Hot Encoding.

**Файлы:**
- [Входные данные](exercises/data_preprocessing_missing_encoding/input.csv)
- [Условие задачи](exercises/data_preprocessing_missing_encoding/task.md)
- [Ответ](exercises/data_preprocessing_missing_encoding/answer.csv)

---

### 3. Обработка выбросов (Outliers Handling)

**Описание:** Определение и удаление выбросов с использованием метода IQR (Interquartile Range).

**Файлы:**
- [Входные данные](exercises/outliers_handling/input.csv)
- [Условие задачи](exercises/outliers_handling/task.md)
- [Ответ](exercises/outliers_handling/answer.csv)

---

### 4. Кодирование категориальных признаков (Categorical Encoding)

**Описание:** Применение различных методов кодирования: Label Encoding, One-Hot Encoding и Target Encoding.

**Файлы:**
- [Входные данные](exercises/categorical_encoding/input.csv)
- [Условие задачи](exercises/categorical_encoding/task.md)
- [Ответ](exercises/categorical_encoding/answer.csv)

---

### 5. Масштабирование и нормализация (Scaling and Normalization)

**Описание:** Применение различных методов масштабирования: MinMaxScaler, StandardScaler, RobustScaler и MaxAbsScaler.

**Файлы:**
- [Входные данные](exercises/scaling_normalization/input.csv)
- [Условие задачи](exercises/scaling_normalization/task.md)
- [Ответ](exercises/scaling_normalization/answer.csv)
- [Решение с комментариями](exercises/scaling_normalization/solution.py) ✨ NEW!

---

### 6. Выбор признаков (Feature Selection) - Цикл упражнений

**Описание:** Комплексный цикл упражнений по различным методам отбора признаков (Filter, Wrapper, Embedded).

**Упражнения:**

#### 6.1. Variance Threshold (Удаление низкодисперсных признаков)
- [Входные данные](exercises/feature_selection/exercise1_variance_threshold/input.csv)
- [Условие задачи](exercises/feature_selection/exercise1_variance_threshold/task.md)
- [Ответ](exercises/feature_selection/exercise1_variance_threshold/answer.csv)

#### 6.2. Mutual Information (Взаимная информация)
- [Входные данные](exercises/feature_selection/exercise2_mutual_information/input.csv)
- [Условие задачи](exercises/feature_selection/exercise2_mutual_information/task.md)
- [Ответ](exercises/feature_selection/exercise2_mutual_information/answer.csv)

#### 6.3. RFE - Recursive Feature Elimination
- [Входные данные](exercises/feature_selection/exercise3_rfe/input.csv)
- [Условие задачи](exercises/feature_selection/exercise3_rfe/task.md)
- [Ответ](exercises/feature_selection/exercise3_rfe/answer.csv)

#### 6.4. Lasso (L1 регуляризация)
- [Входные данные](exercises/feature_selection/exercise4_lasso/input.csv)
- [Условие задачи](exercises/feature_selection/exercise4_lasso/task.md)
- [Ответ](exercises/feature_selection/exercise4_lasso/answer.csv)

#### 6.5. Tree-based Feature Importance
- [Входные данные](exercises/feature_selection/exercise5_tree_importance/input.csv)
- [Условие задачи](exercises/feature_selection/exercise5_tree_importance/task.md)
- [Ответ](exercises/feature_selection/exercise5_tree_importance/answer.csv)

#### 6.6. Удаление коррелирующих признаков
- [Входные данные](exercises/feature_selection/exercise6_remove_correlated/input.csv)
- [Условие задачи](exercises/feature_selection/exercise6_remove_correlated/task.md)
- [Ответ](exercises/feature_selection/exercise6_remove_correlated/answer.csv)

#### 6.7. Корреляция с целевой переменной
- [Входные данные](exercises/feature_selection/exercise7_correlation_target/input.csv)
- [Условие задачи](exercises/feature_selection/exercise7_correlation_target/task.md)
- [Ответ](exercises/feature_selection/exercise7_correlation_target/answer.csv)

**См. также:** [README цикла упражнений](exercises/feature_selection/README.md)

---

### 7. Снижение размерности (Dimensionality Reduction)

**Описание:** Применение метода PCA (Principal Component Analysis) для снижения размерности данных.

**Файлы:**
- [Входные данные](exercises/dimensionality_reduction/input.csv)
- [Условие задачи](exercises/dimensionality_reduction/task.md)
- [Ответ](exercises/dimensionality_reduction/answer.csv)

---

### 8. Балансировка классов (Imbalanced Data)

**Описание:** Балансировка несбалансированных данных с использованием метода Random Oversampling.

**Файлы:**
- [Входные данные](exercises/imbalanced_data/input.csv)
- [Условие задачи](exercises/imbalanced_data/task.md)
- [Ответ](exercises/imbalanced_data/answer.csv)

## Базовые алгоритмы ML

> Упражнения к этапам 2, 4–6 [Пути новичка](ROADMAP.md). Данные и эталоны создаёт `tools/generate_basic_exercises.py`.

### 9. Метрики регрессии

**Теория:** [Метрики регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/regression_metrics_cheatsheet.html)

**Описание:** MAE, MSE, RMSE, R² и MAPE — вручную и через sklearn.metrics.

**Файлы:**
- [Входные данные](exercises/regression_metrics/input.csv)
- [Условие задачи](exercises/regression_metrics/task.md)
- [Ответ](exercises/regression_metrics/answer.csv)
- [Решение с комментариями](exercises/regression_metrics/solution.py)

### 10. Метрики классификации

**Теория:** [Метрики классификации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/classification_metrics_cheatsheet.html)

**Описание:** Матрица ошибок, accuracy, precision, recall, F1 и ROC-AUC.

**Файлы:**
- [Входные данные](exercises/classification_metrics/input.csv)
- [Условие задачи](exercises/classification_metrics/task.md)
- [Ответ](exercises/classification_metrics/answer.csv)
- [Решение с комментариями](exercises/classification_metrics/solution.py)

### 11. Линейная регрессия: цена квартиры

**Теория:** [Линейная регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/linreg_cheatsheet.html)

**Описание:** Обучение LinearRegression на train, предсказание на test, разбор коэффициентов.

**Файлы:**
- [Входные данные](exercises/linreg/input.csv)
- [Условие задачи](exercises/linreg/task.md)
- [Ответ](exercises/linreg/answer.csv)
- [Решение с комментариями](exercises/linreg/solution.py)

### 12. Логистическая регрессия: отток клиентов

**Теория:** [Логистическая регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/logreg_cheatsheet.html)

**Описание:** StandardScaler без утечки данных, LogisticRegression, вероятности и порог.

**Файлы:**
- [Входные данные](exercises/logreg/input.csv)
- [Условие задачи](exercises/logreg/task.md)
- [Ответ](exercises/logreg/answer.csv)
- [Решение с комментариями](exercises/logreg/solution.py)

### 13. Дерево решений: одобрение кредита

**Теория:** [Деревья решений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/decision_trees_cheatsheet.html)

**Описание:** DecisionTreeClassifier с ограничением глубины, правила дерева, важность признаков, переобучение.

**Файлы:**
- [Входные данные](exercises/decision_trees/input.csv)
- [Условие задачи](exercises/decision_trees/task.md)
- [Ответ](exercises/decision_trees/answer.csv)
- [Решение с комментариями](exercises/decision_trees/solution.py)

### 14. Кросс-валидация

**Теория:** [Кросс-валидация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cross_validation_cheatsheet.html)

**Описание:** KFold вручную и через cross_val_score, R² по фолдам.

**Файлы:**
- [Входные данные](exercises/cross_validation/input.csv)
- [Условие задачи](exercises/cross_validation/task.md)
- [Ответ](exercises/cross_validation/answer.csv)
- [Решение с комментариями](exercises/cross_validation/solution.py)

### 15. Градиентный бустинг: прокат велосипедов

**Теория:** [Градиентный бустинг для регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gradient_boosting_regression_cheatsheet.html)

**Описание:** GradientBoostingRegressor на нелинейных данных, сравнение с линейной моделью.

**Файлы:**
- [Входные данные](exercises/gradient_boosting_regression/input.csv)
- [Условие задачи](exercises/gradient_boosting_regression/task.md)
- [Ответ](exercises/gradient_boosting_regression/answer.csv)
- [Решение с комментариями](exercises/gradient_boosting_regression/solution.py)

## Основы Deep Learning (NumPy)

> Упражнения к Пути 2: всё, что фреймворки делают внутри, посчитано вручную на NumPy — PyTorch не нужен. Данные и эталоны создаёт `tools/generate_dl_exercises.py`.

### 16. Функции активации и их производные

**Теория:** [Функции активации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/activation_functions_cheatsheet.html)

**Описание:** sigmoid, tanh, ReLU и их производные; почему сигмоида ведёт к затуханию градиента.

**Файлы:**
- [Входные данные](exercises/activation_functions/input.csv)
- [Условие задачи](exercises/activation_functions/task.md)
- [Ответ](exercises/activation_functions/answer.csv)
- [Решение с комментариями](exercises/activation_functions/solution.py)
- [Тетрадка Jupyter](exercises/activation_functions/exercise.ipynb)

### 17. Softmax и кросс-энтропия

**Теория:** [Функции потерь](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/loss_functions_cheatsheet.html)

**Описание:** Численно устойчивые softmax и log-sum-exp на логитах ±1000.

**Файлы:**
- [Входные данные](exercises/loss_functions/input.csv)
- [Условие задачи](exercises/loss_functions/task.md)
- [Ответ](exercises/loss_functions/answer.csv)
- [Решение с комментариями](exercises/loss_functions/solution.py)
- [Тетрадка Jupyter](exercises/loss_functions/exercise.ipynb)

### 18. Обратное распространение для одного нейрона

**Теория:** [Обратное распространение ошибки](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/backpropagation_cheatsheet.html)

**Описание:** Прямой и обратный проход по цепному правилу, проверка градиента численно.

**Файлы:**
- [Входные данные](exercises/backpropagation/input.csv)
- [Условие задачи](exercises/backpropagation/task.md)
- [Ответ](exercises/backpropagation/answer.csv)
- [Решение с комментариями](exercises/backpropagation/solution.py)
- [Тетрадка Jupyter](exercises/backpropagation/exercise.ipynb)

### 19. Прямой проход MLP

**Теория:** [Архитектура MLP](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mlp_cheatsheet.html)

**Описание:** Сеть 2 → 3 → 1 с весами из weights.json: матричный прямой проход.

**Файлы:**
- [Входные данные](exercises/mlp/input.csv)
- [Веса сети](exercises/mlp/weights.json)
- [Условие задачи](exercises/mlp/task.md)
- [Ответ](exercises/mlp/answer.csv)
- [Решение с комментариями](exercises/mlp/solution.py)
- [Тетрадка Jupyter](exercises/mlp/exercise.ipynb)

### 20. SGD, Momentum и Adam

**Теория:** [Оптимизаторы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optimizers_cheatsheet.html)

**Описание:** Три оптимизатора на функции-«овраге» по формулам, 50 шагов.

**Файлы:**
- [Входные данные](exercises/optimizers/input.csv)
- [Условие задачи](exercises/optimizers/task.md)
- [Ответ](exercises/optimizers/answer.csv)
- [Решение с комментариями](exercises/optimizers/solution.py)
- [Тетрадка Jupyter](exercises/optimizers/exercise.ipynb)

### 21. Свёртка, ReLU и max pooling

**Теория:** [Основы свёртки](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cnn_basics_cheatsheet.html)

**Описание:** Детектор вертикальной границы на изображении 7×7, затем ReLU и пулинг 2×2.

**Файлы:**
- [Входные данные](exercises/cnn_basics/input.csv)
- [Условие задачи](exercises/cnn_basics/task.md)
- [Ответ](exercises/cnn_basics/answer.csv)
- [Решение с комментариями](exercises/cnn_basics/solution.py)
- [Тетрадка Jupyter](exercises/cnn_basics/exercise.ipynb)

### 22. Scaled dot-product attention

**Теория:** [Self-attention](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/self_attention_cheatsheet.html)

**Описание:** softmax(QKᵀ/√d)·V для 4 токенов; бонус — каузальная маска.

**Файлы:**
- [Входные данные](exercises/self_attention/input.csv)
- [Условие задачи](exercises/self_attention/task.md)
- [Ответ](exercises/self_attention/answer.csv)
- [Решение с комментариями](exercises/self_attention/solution.py)
- [Тетрадка Jupyter](exercises/self_attention/exercise.ipynb)



---

## 📚 Как работать с упражнениями

### Рекомендуемый подход (для максимального обучения)

#### Шаг 1: Изучите теорию
Перед решением упражнения обязательно прочитайте соответствующий cheatsheet:
- Упражнение 1 → [Preprocessing cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/preprocessing_cheatsheet.html)
- Упражнение 2 → [Data Preprocessing cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
- И так далее (ссылки в конце документа)

#### Шаг 2: Попробуйте решить самостоятельно
1. Перейдите в папку упражнения: `cd exercises/имя_упражнения/`
2. Прочитайте условие в `task.md`
3. Изучите данные в `input.csv`
4. Напишите свой код и сохраните результат в `my_answer.csv` (не перезаписывайте `answer.csv` — это эталон)
5. Проверьте результат автоматически (из корня репозитория):

```bash
python tools/check_exercise.py exercises/preprocessing
```

Скрипт покажет недостающие столбцы и строки с неверными значениями. Список всех упражнений: `python tools/check_exercise.py --list`. Условия и файлы удобно открывать в учебном портале (`python start.py`, вкладка «Практика»).

**Удобнее в Jupyter:** в каждой папке упражнения лежит тетрадка `exercise.ipynb` — условие, загрузка `input.csv`, ячейка для решения и ячейка проверки. Запишите итог в переменную `result` и запустите последнюю ячейку: она сохранит `my_answer.csv` и сравнит его с эталоном.

```bash
jupyter notebook exercises/linreg/exercise.ipynb
```

#### Шаг 3: Проверьте решение
- Если получилось - отлично! Отметьте упражнение решённым на портале и переходите к следующему
- Если не получилось - изучите `solution.py` (где доступно)
- Экспериментируйте с параметрами

### Способ 1: Самостоятельное решение (рекомендуется)

```bash
cd exercises/preprocessing/
# Прочитайте task.md
# Изучите input.csv
# Напишите свой код в новом файле my_solution.py
python my_solution.py
# Сравните результат с answer.csv
```

**Когда использовать:** Вы изучили cheatsheet и хотите проверить понимание.

### Способ 2: Изучение через готовое решение

```bash
cd exercises/имя_упражнения/
# Откройте solution.py в редакторе
# Изучите код построчно с комментариями
python solution.py
# Экспериментируйте: меняйте параметры и смотрите результат
```

**Когда использовать:** Упражнение оказалось сложным или нужны подсказки.

### Способ 3: Интерактивная работа в Jupyter

```bash
# Установите jupyter если нужно
pip install jupyter

# Запустите Jupyter
jupyter notebook

# Создайте новый notebook в папке упражнения
# Копируйте код из solution.py по блокам
# Выполняйте и изучайте каждый блок отдельно
```

**Когда использовать:** Хотите экспериментировать и визуализировать промежуточные результаты.

### 🎓 Советы по решению упражнений

✅ **Решайте по порядку** - упражнения расположены по возрастанию сложности  
✅ **Не спешите** - лучше разобраться с одним упражнением, чем пробежаться по всем  
✅ **Экспериментируйте** - меняйте параметры, смотрите что произойдет  
✅ **Ведите заметки** - записывайте что узнали и что не получилось  
✅ **Используйте отладку** - добавляйте print() для понимания процесса

## Требования

Для выполнения упражнений потребуются следующие Python библиотеки:
- pandas
- numpy
- scikit-learn

Установка:
```bash
pip install pandas numpy scikit-learn
```

## Связанные чит-шиты

- [Предобработка данных (Preprocessing)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/preprocessing_cheatsheet.html)
- [Data Preprocessing (missing values, encoding)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
- [Обработка выбросов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/outliers_handling_cheatsheet.html)
- [Кодирование категориальных признаков](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/categorical_encoding_cheatsheet.html)
- [Масштабирование и нормализация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/scaling_normalization_cheatsheet.html)
- [Выбор признаков (Feature Selection)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_selection_cheatsheet.html)
- [Снижение размерности (Dimensionality Reduction)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dimensionality_reduction_cheatsheet.html)
- [Балансировка классов (Imbalanced Data)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imbalanced_data_cheatsheet.html)

---

## 🔗 Навигация по курсу

### 📚 Основные разделы
- [🏠 Главная страница](README.md) - полный список cheatsheets
- [🗺️ Дорожная карта](ROADMAP.md) - структурированные пути обучения
- [🎥 Анимации](animations/README.md) - визуализации алгоритмов

### 🛠️ Для разработчиков
- [📋 Setup Guide](SETUP.md) - инструкции по установке
- [🤝 Contributing](CONTRIBUTING.md) - как внести вклад

### 💡 Нужна помощь?
- Возникли вопросы? [Создайте Issue](https://github.com/gurovic/MLCheatSheets/issues)
- Хотите обсудить? Загляните в [Discussions](https://github.com/gurovic/MLCheatSheets/discussions)

---

<div align="center">

**Практика делает совершенным! 💪**

© 2024 MLCheatSheets. Автор: Владимир Гуровиц (школа "Летово")

[⬆ Наверх](#-упражнения-по-машинному-обучению)

</div>
