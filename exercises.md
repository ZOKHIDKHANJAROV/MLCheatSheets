# 💪 Упражнения по машинному обучению

> **Практические задания для закрепления теории из [cheatsheets](README.md)**

Всего упражнений: **53**, из них 🐞 «Найди ошибку» — 6. Удобнее всего решать их в портале (`python start.py` → вкладка «Практика»): там подсказки открываются по одной, решённое упражнение отмечается само, после решения открывается разбор, а через 2 недели упражнение возвращается на повтор.

## 🎯 Как устроено упражнение

- `input.csv` (и другие файлы данных) — входные данные;
- `task.md` — условие, **подсказки** от лёгкой к сильной, формат ответа и **разбор** (читайте после попытки);
- `exercise.ipynb` — тетрадка с загрузкой данных и ячейкой проверки;
- `solution.py` — эталонное решение с комментариями;
- `answer.key` — эталонный ответ в закодированном виде, чтобы не подсмотреть случайно;
- 🐞 у заданий «Найди ошибку» есть `buggy.py` — правдоподобный код с ошибками, который надо найти и исправить.

Проверка решения (из корня репозитория):

```bash
python tools/check_exercise.py exercises/<папка>      # ваш результат — my_answer.csv в папке упражнения
python tools/check_exercise.py --list                  # все упражнения и что уже решено
```

Проверка не только говорит «неверно», но и подсказывает вероятную причину: порядок строк, округление, ddof, проценты вместо долей, сдвиг нумерации, перепутанные столбцы, лишний индекс.

Кроме расчётных упражнений в портале есть 🧩 **задачи на понимание** ([understanding/](understanding/)): «Предскажи результат» с проверкой в интерактиве и «Выбери подход» — короткие рабочие кейсы.

После упражнений пути попробуйте его **мини-проект** — сквозную задачу без готового ответа: [projects/](projects/README.md).

## Упражнения по учебным путям ([ROADMAP](ROADMAP.md))

### 🌱 Путь 1: Полный новичок в ML (3-6 месяцев)

**Этап 1: Основы Python и библиотек (2-4 недели)**

- **Предобработка данных (Preprocessing)** — [условие](exercises/preprocessing/task.md) · [папка](exercises/preprocessing/) · теория: [Предобработка данных (Preprocessing)](cheatsheets/preprocessing_cheatsheet.html)

**Этап 2: Основы теории ML (3-4 недели)**

- **Кросс-валидация** — [условие](exercises/cross_validation/task.md) · [папка](exercises/cross_validation/) · теория: [Кросс-валидация](cheatsheets/cross_validation_cheatsheet.html)
- **Градиентный спуск для линейной регрессии вручную** — [условие](exercises/gradient_descent/task.md) · [папка](exercises/gradient_descent/) · теория: [Градиентный спуск (Gradient Descent)](cheatsheets/gradient_descent_cheatsheet.html)

**Этап 3: Подготовка данных (2-3 недели)**

- **Обработка пропущенных значений и кодирование** — [условие](exercises/data_preprocessing_missing_encoding/task.md) · [папка](exercises/data_preprocessing_missing_encoding/) · теория: [Data Preprocessing (missing values, encoding)](cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
- **Обработка выбросов: метод IQR** — [условие](exercises/outliers_handling/task.md) · [папка](exercises/outliers_handling/) · теория: [Обработка выбросов](cheatsheets/outliers_handling_cheatsheet.html)
- **Кодирование категориальных признаков** — [условие](exercises/categorical_encoding/task.md) · [папка](exercises/categorical_encoding/) · теория: [Кодирование категориальных признаков](cheatsheets/categorical_encoding_cheatsheet.html)
- **Масштабирование и нормализация** — [условие](exercises/scaling_normalization/task.md) · [папка](exercises/scaling_normalization/) · теория: [Масштабирование и нормализация](cheatsheets/scaling_normalization_cheatsheet.html)
- **Отбор признаков 1: удаление низкодисперсных признаков (Variance Threshold)** — [условие](exercises/feature_selection/exercise1_variance_threshold/task.md) · [папка](exercises/feature_selection/exercise1_variance_threshold/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 2: взаимная информация (Mutual Information)** — [условие](exercises/feature_selection/exercise2_mutual_information/task.md) · [папка](exercises/feature_selection/exercise2_mutual_information/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 3: рекурсивное удаление признаков (RFE)** — [условие](exercises/feature_selection/exercise3_rfe/task.md) · [папка](exercises/feature_selection/exercise3_rfe/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 4: Lasso (L1-регуляризация)** — [условие](exercises/feature_selection/exercise4_lasso/task.md) · [папка](exercises/feature_selection/exercise4_lasso/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 5: важность признаков в случайном лесе** — [условие](exercises/feature_selection/exercise5_tree_importance/task.md) · [папка](exercises/feature_selection/exercise5_tree_importance/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 6: удаление коррелирующих признаков** — [условие](exercises/feature_selection/exercise6_remove_correlated/task.md) · [папка](exercises/feature_selection/exercise6_remove_correlated/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Отбор признаков 7: корреляция с целевой переменной** — [условие](exercises/feature_selection/exercise7_correlation_target/task.md) · [папка](exercises/feature_selection/exercise7_correlation_target/) · теория: [Выбор признаков (Feature Selection)](cheatsheets/feature_selection_cheatsheet.html)
- **Снижение размерности: PCA** — [условие](exercises/dimensionality_reduction/task.md) · [папка](exercises/dimensionality_reduction/) · теория: [Снижение размерности (Dimensionality Reduction)](cheatsheets/dimensionality_reduction_cheatsheet.html)
- **Балансировка классов: random oversampling** — [условие](exercises/imbalanced_data/task.md) · [папка](exercises/imbalanced_data/) · теория: [Балансировка классов (Imbalanced Data)](cheatsheets/imbalanced_data_cheatsheet.html)
- 🐞 **Найди ошибку: масштабирование до разбиения** — [условие](exercises/bug_scaler_leakage/task.md) · [папка](exercises/bug_scaler_leakage/) · теория: [Масштабирование и нормализация](cheatsheets/scaling_normalization_cheatsheet.html)
- 🐞 **Найди ошибку: утечка целевой переменной** — [условие](exercises/bug_target_leakage/task.md) · [папка](exercises/bug_target_leakage/) · теория: [Feature Engineering](cheatsheets/feature_engineering_cheatsheet.html)

**Этап 4: Классические алгоритмы - Регрессия (2-3 недели)**

- **Линейная регрессия: цена квартиры** — [условие](exercises/linreg/task.md) · [папка](exercises/linreg/) · теория: [Линейная регрессия](cheatsheets/linreg_cheatsheet.html)
- **Метрики регрессии** — [условие](exercises/regression_metrics/task.md) · [папка](exercises/regression_metrics/) · теория: [Метрики регрессии](cheatsheets/regression_metrics_cheatsheet.html)

**Этап 5: Классические алгоритмы - Классификация (2-3 недели)**

- **Логистическая регрессия: отток клиентов** — [условие](exercises/logreg/task.md) · [папка](exercises/logreg/) · теория: [Логистическая регрессия](cheatsheets/logreg_cheatsheet.html)
- **Метрики классификации** — [условие](exercises/classification_metrics/task.md) · [папка](exercises/classification_metrics/) · теория: [Метрики классификации](cheatsheets/classification_metrics_cheatsheet.html)
- **ROC-кривая и AUC вручную** — [условие](exercises/roc_auc/task.md) · [папка](exercises/roc_auc/) · теория: [ROC и AUC](cheatsheets/roc_auc_cheatsheet.html)
- **k ближайших соседей вручную** — [условие](exercises/knn/task.md) · [папка](exercises/knn/) · теория: [k-ближайших соседей (k-NN)](cheatsheets/knn_cheatsheet.html)
- **Мультиномиальный наивный Байес для спама** — [условие](exercises/naive_bayes/task.md) · [папка](exercises/naive_bayes/) · теория: [Наивный Байес (Naive Bayes)](cheatsheets/naive_bayes_cheatsheet.html)
- 🐞 **Найди ошибку: метрики классификации** — [условие](exercises/bug_metrics/task.md) · [папка](exercises/bug_metrics/) · теория: [Метрики классификации](cheatsheets/classification_metrics_cheatsheet.html)

**Этап 6: Деревья и ансамбли (3-4 недели)**

- **Дерево решений: одобрение кредита** — [условие](exercises/decision_trees/task.md) · [папка](exercises/decision_trees/) · теория: [Деревья решений для классификации](cheatsheets/decision_trees_cheatsheet.html)
- **Случайный лес: OOB-оценка и важность признаков** — [условие](exercises/random_forest/task.md) · [папка](exercises/random_forest/) · теория: [Случайный лес для классификации](cheatsheets/random_forest_cheatsheet.html)
- **Градиентный бустинг: прокат велосипедов** — [условие](exercises/gradient_boosting_regression/task.md) · [папка](exercises/gradient_boosting_regression/) · теория: [Градиентный бустинг для регрессии](cheatsheets/gradient_boosting_regression_cheatsheet.html)

### 🚀 Путь 2: Специализация в глубоком обучении (4-8 месяцев)

**Этап 1: Основы нейросетей (3-4 недели)**

- **Функции активации и их производные** — [условие](exercises/activation_functions/task.md) · [папка](exercises/activation_functions/) · теория: [Функции активации (ReLU, Sigmoid, Tanh, Softmax)](cheatsheets/activation_functions_cheatsheet.html)
- **Softmax и кросс-энтропия (численно устойчиво)** — [условие](exercises/loss_functions/task.md) · [папка](exercises/loss_functions/) · теория: [Функции потерь для классификации и регрессии](cheatsheets/loss_functions_cheatsheet.html)
- **Обратное распространение для одного нейрона** — [условие](exercises/backpropagation/task.md) · [папка](exercises/backpropagation/) · теория: [Обратное распространение ошибки (Backpropagation)](cheatsheets/backpropagation_cheatsheet.html)
- **Прямой проход многослойного перцептрона** — [условие](exercises/mlp/task.md) · [папка](exercises/mlp/) · теория: [Архитектура MLP (многослойный перцептрон)](cheatsheets/mlp_cheatsheet.html)
- **SGD, Momentum и Adam на «овраге»** — [условие](exercises/optimizers/task.md) · [папка](exercises/optimizers/) · теория: [Оптимизаторы (SGD, Momentum, Adam, RMSProp)](cheatsheets/optimizers_cheatsheet.html)
- 🐞 **Найди ошибку: цикл градиентного спуска** — [условие](exercises/bug_training_loop/task.md) · [папка](exercises/bug_training_loop/) · теория: [Градиентный спуск (Gradient Descent)](cheatsheets/gradient_descent_cheatsheet.html)

**Этап 2: Фреймворки (2-3 недели)**

- **Цикл обучения как в PyTorch — на NumPy** — [условие](exercises/pytorch_full_guide/task.md) · [папка](exercises/pytorch_full_guide/) · теория: [PyTorch полный гайд](cheatsheets/pytorch_full_guide_cheatsheet.html)

**Этап 3: Сверточные сети (Computer Vision) (3-4 недели)**

- **Свёртка, ReLU и max pooling вручную** — [условие](exercises/cnn_basics/task.md) · [папка](exercises/cnn_basics/) · теория: [Основы свертки](cheatsheets/cnn_basics_cheatsheet.html)
- **IoU и Non-Maximum Suppression** — [условие](exercises/object_detection/task.md) · [папка](exercises/object_detection/) · теория: [Детекция объектов - Faster R-CNN, SSD, RetinaNet, EfficientDet](cheatsheets/object_detection_cheatsheet.html)
- **Метрики сегментации: IoU, Dice, pixel accuracy** — [условие](exercises/image_segmentation/task.md) · [папка](exercises/image_segmentation/) · теория: [Сегментация изображений (U-Net, DeepLab, Mask R-CNN)](cheatsheets/image_segmentation_cheatsheet.html)

**Этап 4: Рекуррентные сети и NLP (3-4 недели)**

- **Ячейка LSTM: прямой проход по последовательности** — [условие](exercises/lstm/task.md) · [папка](exercises/lstm/) · теория: [LSTM (Long Short-Term Memory)](cheatsheets/lstm_cheatsheet.html)
- **Word embeddings: косинусная близость и аналогии** — [условие](exercises/word_embeddings/task.md) · [папка](exercises/word_embeddings/) · теория: [Векторные представления слов (Word2Vec, GloVe, FastText)](cheatsheets/word_embeddings_cheatsheet.html)

**Этап 5: Трансформеры и LLM (4-5 недель)**

- **Scaled dot-product attention** — [условие](exercises/self_attention/task.md) · [папка](exercises/self_attention/) · теория: [Self-attention](cheatsheets/self_attention_cheatsheet.html)

**Этап 6: Генеративные модели (3-4 недели)**

- **VAE: репараметризация, KL и ELBO** — [условие](exercises/vae/task.md) · [папка](exercises/vae/) · теория: [Вариационные автоэнкодеры (VAE)](cheatsheets/vae_cheatsheet.html)
- **Функции потерь GAN: minimax и non-saturating** — [условие](exercises/gan/task.md) · [папка](exercises/gan/) · теория: [Генеративно-состязательные сети (GAN)](cheatsheets/gan_cheatsheet.html)

### 💼 Путь 3: ML для работы (Production ML) (2-4 месяца)

**Этап 1: Оптимизация моделей (2-3 недели)**

- **Подбор глубины дерева кросс-валидацией** — [условие](exercises/hyperparameter_tuning/task.md) · [папка](exercises/hyperparameter_tuning/) · теория: [Grid Search / Random Search](cheatsheets/hyperparameter_tuning_cheatsheet.html)

**Этап 2: Интерпретируемость (2 недели)**

- **Значения Шепли вручную** — [условие](exercises/shap/task.md) · [папка](exercises/shap/) · теория: [SHAP значения](cheatsheets/shap_cheatsheet.html)

**Этап 3: Production и MLOps (3-4 недели)**

- **Утечка данных: отбор признаков вне пайплайна** — [условие](exercises/ml_pipelines/task.md) · [папка](exercises/ml_pipelines/) · теория: [Пайплайны ML](cheatsheets/ml_pipelines_cheatsheet.html)
- 🐞 **Найди ошибку: отбор признаков до кросс-валидации** — [условие](exercises/bug_cv_selection/task.md) · [папка](exercises/bug_cv_selection/) · теория: [Пайплайны ML](cheatsheets/ml_pipelines_cheatsheet.html)

**Этап 4: Специальные темы (2-3 недели)**

- **Дрейф данных: Population Stability Index** — [условие](exercises/concept_drift/task.md) · [папка](exercises/concept_drift/) · теория: [Дрейф данных и концептуальный дрейф](cheatsheets/concept_drift_cheatsheet.html)

### 📈 Путь 4: Специализация по доменам

**Computer Vision специалист**

- **IoU и Non-Maximum Suppression** — [условие](exercises/object_detection/task.md) · [папка](exercises/object_detection/) · теория: [Детекция объектов - Faster R-CNN, SSD, RetinaNet, EfficientDet](cheatsheets/object_detection_cheatsheet.html)
- **Метрики сегментации: IoU, Dice, pixel accuracy** — [условие](exercises/image_segmentation/task.md) · [папка](exercises/image_segmentation/) · теория: [Сегментация изображений (U-Net, DeepLab, Mask R-CNN)](cheatsheets/image_segmentation_cheatsheet.html)

**NLP специалист**

- **TF-IDF вручную** — [условие](exercises/tfidf/task.md) · [папка](exercises/tfidf/) · теория: [TF-IDF и векторное пространство](cheatsheets/tfidf_cheatsheet.html)

**Time Series специалист**

- **Экспоненциальное сглаживание: SES и Хольт** — [условие](exercises/exponential_smoothing/task.md) · [папка](exercises/exponential_smoothing/) · теория: [Экспоненциальное сглаживание](cheatsheets/exponential_smoothing_cheatsheet.html)
- 🐞 **Найди ошибку: валидация временного ряда** — [условие](exercises/bug_time_series_split/task.md) · [папка](exercises/bug_time_series_split/) · теория: [Валидация временных рядов](cheatsheets/time_series_validation_cheatsheet.html)

**Reinforcement Learning специалист**

- **Q-learning и SARSA по записанной траектории** — [условие](exercises/q_learning_sarsa/task.md) · [папка](exercises/q_learning_sarsa/) · теория: [Q-learning и SARSA](cheatsheets/q_learning_sarsa_cheatsheet.html)

## Требования

Для выполнения упражнений потребуются следующие Python библиотеки:
- pandas
- numpy
- scikit-learn

Установка:
```bash
pip install pandas numpy scikit-learn
```

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
