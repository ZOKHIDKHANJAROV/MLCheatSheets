# 💪 Упражнения по машинному обучению

> **Практические задания для закрепления теории из [cheatsheets](README.md)**

Всего упражнений: **127**, из них 🐞 «Найди ошибку» — 6. Удобнее всего решать их в портале (`python start.py` → вкладка «Практика»): там подсказки открываются по одной, решённое упражнение отмечается само, после решения открывается разбор, а через 2 недели упражнение возвращается на повтор.

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
- **NumPy: оси, broadcasting, маски и матричное умножение** — [условие](exercises/numpy_ml/task.md) · [папка](exercises/numpy_ml/) · теория: [NumPy для ML](cheatsheets/numpy_ml_cheatsheet.html)
- **Pandas: чистка, пропуски по группам, groupby и merge** — [условие](exercises/pandas_ml/task.md) · [папка](exercises/pandas_ml/) · теория: [Pandas для ML](cheatsheets/pandas_ml_cheatsheet.html)
- **Что показывают ящик с усами и гистограмма: считаем вручную** — [условие](exercises/matplotlib_seaborn_ml/task.md) · [папка](exercises/matplotlib_seaborn_ml/) · теория: [Matplotlib/Seaborn для визуализации](cheatsheets/matplotlib_seaborn_ml_cheatsheet.html)

**Этап 2: Основы теории ML (3-4 недели)**

- **Кросс-валидация** — [условие](exercises/cross_validation/task.md) · [папка](exercises/cross_validation/) · теория: [Кросс-валидация](cheatsheets/cross_validation_cheatsheet.html)
- **Градиентный спуск для линейной регрессии вручную** — [условие](exercises/gradient_descent/task.md) · [папка](exercises/gradient_descent/) · теория: [Градиентный спуск (Gradient Descent)](cheatsheets/gradient_descent_cheatsheet.html)
- **Разложение ошибки на смещение и дисперсию** — [условие](exercises/bias_variance/task.md) · [папка](exercises/bias_variance/) · теория: [Смещение-дисперсия](cheatsheets/bias_variance_cheatsheet.html)
- **Кривые обучения: недообучение и переобучение** — [условие](exercises/overfitting_underfitting/task.md) · [папка](exercises/overfitting_underfitting/) · теория: [Переобучение и недообучение](cheatsheets/overfitting_underfitting_cheatsheet.html)

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
- **Полиномиальная регрессия: выбор степени по кросс-валидации** — [условие](exercises/polynomial_regression/task.md) · [папка](exercises/polynomial_regression/) · теория: [Полиномиальная регрессия](cheatsheets/polynomial_regression_cheatsheet.html)
- **Ridge, Lasso и ElasticNet: путь регуляризации** — [условие](exercises/ridge_lasso_elasticnet/task.md) · [папка](exercises/ridge_lasso_elasticnet/) · теория: [Ридж, Лассо, ElasticNet](cheatsheets/ridge_lasso_elasticnet_cheatsheet.html)

**Этап 5: Классические алгоритмы - Классификация (2-3 недели)**

- **Логистическая регрессия: отток клиентов** — [условие](exercises/logreg/task.md) · [папка](exercises/logreg/) · теория: [Логистическая регрессия](cheatsheets/logreg_cheatsheet.html)
- **Метрики классификации** — [условие](exercises/classification_metrics/task.md) · [папка](exercises/classification_metrics/) · теория: [Метрики классификации](cheatsheets/classification_metrics_cheatsheet.html)
- **ROC-кривая и AUC вручную** — [условие](exercises/roc_auc/task.md) · [папка](exercises/roc_auc/) · теория: [ROC и AUC](cheatsheets/roc_auc_cheatsheet.html)
- **k ближайших соседей вручную** — [условие](exercises/knn/task.md) · [папка](exercises/knn/) · теория: [k-ближайших соседей (k-NN)](cheatsheets/knn_cheatsheet.html)
- **Мультиномиальный наивный Байес для спама** — [условие](exercises/naive_bayes/task.md) · [папка](exercises/naive_bayes/) · теория: [Наивный Байес (Naive Bayes)](cheatsheets/naive_bayes_cheatsheet.html)
- 🐞 **Найди ошибку: метрики классификации** — [условие](exercises/bug_metrics/task.md) · [папка](exercises/bug_metrics/) · теория: [Метрики классификации](cheatsheets/classification_metrics_cheatsheet.html)
- **Многоклассовая матрица ошибок вручную** — [условие](exercises/confusion_matrix/task.md) · [папка](exercises/confusion_matrix/) · теория: [Матрица ошибок](cheatsheets/confusion_matrix_cheatsheet.html)

**Этап 6: Деревья и ансамбли (3-4 недели)**

- **Дерево решений: одобрение кредита** — [условие](exercises/decision_trees/task.md) · [папка](exercises/decision_trees/) · теория: [Деревья решений для классификации](cheatsheets/decision_trees_cheatsheet.html)
- **Случайный лес: OOB-оценка и важность признаков** — [условие](exercises/random_forest/task.md) · [папка](exercises/random_forest/) · теория: [Случайный лес для классификации](cheatsheets/random_forest_cheatsheet.html)
- **Градиентный бустинг: прокат велосипедов** — [условие](exercises/gradient_boosting_regression/task.md) · [папка](exercises/gradient_boosting_regression/) · теория: [Градиентный бустинг для регрессии](cheatsheets/gradient_boosting_regression_cheatsheet.html)
- **Градиентный бустинг для классификации вручную** — [условие](exercises/boosting/task.md) · [папка](exercises/boosting/) · теория: [Градиентный бустинг](cheatsheets/boosting_cheatsheet.html)
- **CatBoost вручную: ordered target statistics** — [условие](exercises/catboost/task.md) · [папка](exercises/catboost/) · теория: [CatBoost](cheatsheets/catboost_cheatsheet.html)
- **LightGBM вручную: гистограммы и поиск разбиения по корзинам** — [условие](exercises/lightgbm/task.md) · [папка](exercises/lightgbm/) · теория: [LightGBM](cheatsheets/lightgbm_cheatsheet.html)
- **XGBoost вручную: градиенты, гессианы, прирост и веса листьев** — [условие](exercises/xgboost/task.md) · [папка](exercises/xgboost/) · теория: [XGBoost](cheatsheets/xgboost_cheatsheet.html)

### 🚀 Путь 2: Специализация в глубоком обучении (4-8 месяцев)

**Этап 1: Основы нейросетей (3-4 недели)**

- **Функции активации и их производные** — [условие](exercises/activation_functions/task.md) · [папка](exercises/activation_functions/) · теория: [Функции активации (ReLU, Sigmoid, Tanh, Softmax)](cheatsheets/activation_functions_cheatsheet.html)
- **Softmax и кросс-энтропия (численно устойчиво)** — [условие](exercises/loss_functions/task.md) · [папка](exercises/loss_functions/) · теория: [Функции потерь для классификации и регрессии](cheatsheets/loss_functions_cheatsheet.html)
- **Обратное распространение для одного нейрона** — [условие](exercises/backpropagation/task.md) · [папка](exercises/backpropagation/) · теория: [Обратное распространение ошибки (Backpropagation)](cheatsheets/backpropagation_cheatsheet.html)
- **Прямой проход многослойного перцептрона** — [условие](exercises/mlp/task.md) · [папка](exercises/mlp/) · теория: [Архитектура MLP (многослойный перцептрон)](cheatsheets/mlp_cheatsheet.html)
- **SGD, Momentum и Adam на «овраге»** — [условие](exercises/optimizers/task.md) · [папка](exercises/optimizers/) · теория: [Оптимизаторы (SGD, Momentum, Adam, RMSProp)](cheatsheets/optimizers_cheatsheet.html)
- 🐞 **Найди ошибку: цикл градиентного спуска** — [условие](exercises/bug_training_loop/task.md) · [папка](exercises/bug_training_loop/) · теория: [Градиентный спуск (Gradient Descent)](cheatsheets/gradient_descent_cheatsheet.html)
- **Обучение перцептрона Розенблатта вручную** — [условие](exercises/perceptron/task.md) · [папка](exercises/perceptron/) · теория: [Искусственный нейрон (перцептрон)](cheatsheets/perceptron_cheatsheet.html)

**Этап 2: Фреймворки (2-3 недели)**

- **Цикл обучения как в PyTorch — на NumPy** — [условие](exercises/pytorch_full_guide/task.md) · [папка](exercises/pytorch_full_guide/) · теория: [PyTorch полный гайд](cheatsheets/pytorch_full_guide_cheatsheet.html)
- **Trainer Lightning на NumPy: логирование, EarlyStopping и ModelCheckpoint** — [условие](exercises/pytorch_lightning/task.md) · [папка](exercises/pytorch_lightning/) · теория: [PyTorch Lightning](cheatsheets/pytorch_lightning_cheatsheet.html)
- **Колбэки Keras: EarlyStopping и ReduceLROnPlateau по истории обучения** — [условие](exercises/tensorflow_keras_full_guide/task.md) · [папка](exercises/tensorflow_keras_full_guide/) · теория: [TensorFlow/Keras полный гайд](cheatsheets/tensorflow_keras_full_guide_cheatsheet.html)

**Этап 3: Сверточные сети (Computer Vision) (3-4 недели)**

- **Свёртка, ReLU и max pooling вручную** — [условие](exercises/cnn_basics/task.md) · [папка](exercises/cnn_basics/) · теория: [Основы свертки](cheatsheets/cnn_basics_cheatsheet.html)
- **IoU и Non-Maximum Suppression** — [условие](exercises/object_detection/task.md) · [папка](exercises/object_detection/) · теория: [Детекция объектов - Faster R-CNN, SSD, RetinaNet, EfficientDet](cheatsheets/object_detection_cheatsheet.html)
- **Метрики сегментации: IoU, Dice, pixel accuracy** — [условие](exercises/image_segmentation/task.md) · [папка](exercises/image_segmentation/) · теория: [Сегментация изображений (U-Net, DeepLab, Mask R-CNN)](cheatsheets/image_segmentation_cheatsheet.html)
- **Архитектура CNN: формы выходов и число параметров по слоям** — [условие](exercises/cnn_architectures/task.md) · [папка](exercises/cnn_architectures/) · теория: [Архитектуры CNN](cheatsheets/cnn_architectures_cheatsheet.html)
- **Геометрические аугментации: картинка и рамка синхронно** — [условие](exercises/data_augmentation/task.md) · [папка](exercises/data_augmentation/) · теория: [Data augmentation](cheatsheets/data_augmentation_cheatsheet.html)
- **Max и average pooling со stride и padding** — [условие](exercises/pooling_layers/task.md) · [папка](exercises/pooling_layers/) · теория: [Пулинг-слои](cheatsheets/pooling_layers_cheatsheet.html)
- **Residual-блок ResNet с проекцией в skip-связи** — [условие](exercises/resnet/task.md) · [папка](exercises/resnet/) · теория: [ResNet](cheatsheets/resnet_cheatsheet.html)
- **Transfer learning: замороженный backbone и новая голова** — [условие](exercises/transfer_learning_cnn/task.md) · [папка](exercises/transfer_learning_cnn/) · теория: [Transfer learning для CNN](cheatsheets/transfer_learning_cnn_cheatsheet.html)
- **YOLO: декодирование сетки и якорей, порог уверенности и NMS** — [условие](exercises/yolo/task.md) · [папка](exercises/yolo/) · теория: [Детекция объектов (YOLO)](cheatsheets/yolo_cheatsheet.html)

**Этап 4: Рекуррентные сети и NLP (3-4 недели)**

- **Ячейка LSTM: прямой проход по последовательности** — [условие](exercises/lstm/task.md) · [папка](exercises/lstm/) · теория: [LSTM (Long Short-Term Memory)](cheatsheets/lstm_cheatsheet.html)
- **Word embeddings: косинусная близость и аналогии** — [условие](exercises/word_embeddings/task.md) · [папка](exercises/word_embeddings/) · теория: [Векторные представления слов (Word2Vec, GloVe, FastText)](cheatsheets/word_embeddings_cheatsheet.html)
- **Ячейка GRU: прямой проход в раскладке PyTorch** — [условие](exercises/gru/task.md) · [папка](exercises/gru/) · теория: [GRU](cheatsheets/gru_cheatsheet.html)
- **Простая RNN: прямой проход и BPTT-градиенты** — [условие](exercises/rnn_basics/task.md) · [папка](exercises/rnn_basics/) · теория: [Основы RNN](cheatsheets/rnn_basics_cheatsheet.html)
- **Аддитивное внимание Bahdanau** — [условие](exercises/attention_mechanism/task.md) · [папка](exercises/attention_mechanism/) · теория: [Механизм внимания](cheatsheets/attention_mechanism_cheatsheet.html)

**Этап 5: Трансформеры и LLM (4-5 недель)**

- **Scaled dot-product attention** — [условие](exercises/self_attention/task.md) · [папка](exercises/self_attention/) · теория: [Self-attention](cheatsheets/self_attention_cheatsheet.html)
- **ViT: нарезка на патчи, линейная проекция, [CLS] и позиции** — [условие](exercises/vision_transformers_vit/task.md) · [папка](exercises/vision_transformers_vit/) · теория: [Vision Transformers (ViT)](cheatsheets/vision_transformers_vit_cheatsheet.html)
- **BERT MLM: маскирование 80/10/10 и потери только на масках** — [условие](exercises/bert_language_models/task.md) · [папка](exercises/bert_language_models/) · теория: [BERT и языковые модели](cheatsheets/bert_language_models_cheatsheet.html)
- **Генерация GPT: greedy, temperature, top-k и top-p** — [условие](exercises/gpt_architectures/task.md) · [папка](exercises/gpt_architectures/) · теория: [GPT-архитектуры](cheatsheets/gpt_architectures_cheatsheet.html)
- **Multi-head attention: разрезание на головы, склейка и W_O** — [условие](exercises/multi_head_attention/task.md) · [папка](exercises/multi_head_attention/) · теория: [Multi-head Attention](cheatsheets/multi_head_attention_cheatsheet.html)
- **Оценка вариантов промпта: exact match и self-consistency** — [условие](exercises/prompt_engineering/task.md) · [папка](exercises/prompt_engineering/) · теория: [Prompt Engineering](cheatsheets/prompt_engineering_cheatsheet.html)
- **Позиционное кодирование и блок трансформера: Post-LN и Pre-LN** — [условие](exercises/transformers/task.md) · [папка](exercises/transformers/) · теория: [Архитектура Transformer](cheatsheets/transformers_cheatsheet.html)

**Этап 6: Генеративные модели (3-4 недели)**

- **VAE: репараметризация, KL и ELBO** — [условие](exercises/vae/task.md) · [папка](exercises/vae/) · теория: [Вариационные автоэнкодеры (VAE)](cheatsheets/vae_cheatsheet.html)
- **Функции потерь GAN: minimax и non-saturating** — [условие](exercises/gan/task.md) · [папка](exercises/gan/) · теория: [Генеративно-состязательные сети (GAN)](cheatsheets/gan_cheatsheet.html)
- **Линейный автоэнкодер (≈ PCA) и поиск аномалий по ошибке восстановления** — [условие](exercises/autoencoders/task.md) · [папка](exercises/autoencoders/) · теория: [Автоэнкодеры](cheatsheets/autoencoders_cheatsheet.html)
- **Диффузия: прямой процесс q(x_t | x₀) и шаг обратного DDPM** — [условие](exercises/diffusion_models/task.md) · [папка](exercises/diffusion_models/) · теория: [Diffusion модели](cheatsheets/diffusion_models_cheatsheet.html)

### 💼 Путь 3: ML для работы (Production ML) (2-4 месяца)

**Этап 1: Оптимизация моделей (2-3 недели)**

- **Подбор глубины дерева кросс-валидацией** — [условие](exercises/hyperparameter_tuning/task.md) · [папка](exercises/hyperparameter_tuning/) · теория: [Grid Search / Random Search](cheatsheets/hyperparameter_tuning_cheatsheet.html)
- **MDI против permutation importance вручную** — [условие](exercises/feature_importance/task.md) · [папка](exercises/feature_importance/) · теория: [Важность признаков](cheatsheets/feature_importance_cheatsheet.html)
- **Байесовская оптимизация: гауссовский процесс и Expected Improvement** — [условие](exercises/bayesian_optimization/task.md) · [папка](exercises/bayesian_optimization/) · теория: [Bayesian Optimization](cheatsheets/bayesian_optimization_cheatsheet.html)
- **Один шаг TPE вручную (идея Optuna и Hyperopt)** — [условие](exercises/optuna_hyperopt/task.md) · [папка](exercises/optuna_hyperopt/) · теория: [Optuna / Hyperopt](cheatsheets/optuna_hyperopt_cheatsheet.html)

**Этап 2: Интерпретируемость (2 недели)**

- **Значения Шепли вручную** — [условие](exercises/shap/task.md) · [папка](exercises/shap/) · теория: [SHAP значения](cheatsheets/shap_cheatsheet.html)
- **Частичные зависимости вручную и взаимодействие признаков** — [условие](exercises/partial_dependence/task.md) · [папка](exercises/partial_dependence/) · теория: [Частичные зависимости](cheatsheets/partial_dependence_cheatsheet.html)
- **Permutation importance вручную и сравнение с MDI** — [условие](exercises/explainable_ai_xai/task.md) · [папка](exercises/explainable_ai_xai/) · теория: [Explainable AI (XAI)](cheatsheets/explainable_ai_xai_cheatsheet.html)
- **LIME вручную: взвешенный локальный линейный суррогат** — [условие](exercises/lime/task.md) · [папка](exercises/lime/) · теория: [LIME](cheatsheets/lime_cheatsheet.html)

**Этап 3: Production и MLOps (3-4 недели)**

- **Утечка данных: отбор признаков вне пайплайна** — [условие](exercises/ml_pipelines/task.md) · [папка](exercises/ml_pipelines/) · теория: [Пайплайны ML](cheatsheets/ml_pipelines_cheatsheet.html)
- 🐞 **Найди ошибку: отбор признаков до кросс-валидации** — [условие](exercises/bug_cv_selection/task.md) · [папка](exercises/bug_cv_selection/) · теория: [Пайплайны ML](cheatsheets/ml_pipelines_cheatsheet.html)
- **Quality gate в CI/CD: пропустить модель или остановить конвейер** — [условие](exercises/cicd_ml/task.md) · [папка](exercises/cicd_ml/) · теория: [CI/CD для ML](cheatsheets/cicd_ml_cheatsheet.html)
- **Point-in-time correct join без утечки из будущего** — [условие](exercises/feature_stores/task.md) · [папка](exercises/feature_stores/) · теория: [Feature stores](cheatsheets/feature_stores_cheatsheet.html)
- **Трекинг экспериментов: версия данных по хешу и выбор лучшего запуска** — [условие](exercises/mlops_best_practices/task.md) · [папка](exercises/mlops_best_practices/) · теория: [MLOps Best Practices](cheatsheets/mlops_best_practices_cheatsheet.html)
- **Мониторинг: PSI и KS по признакам и z-тест A/B с доверительным интервалом** — [условие](exercises/model_monitoring_ab_testing/task.md) · [папка](exercises/model_monitoring_ab_testing/) · теория: [Мониторинг моделей и A/B тестирование](cheatsheets/model_monitoring_ab_testing_cheatsheet.html)
- **Сериализация: pipeline, joblib/pickle, SHA-256 и проверка метаданных** — [условие](exercises/model_serialization/task.md) · [папка](exercises/model_serialization/) · теория: [Сериализация моделей](cheatsheets/model_serialization_cheatsheet.html)
- **Задержка инференса: перцентили, ошибки и проверка SLA** — [условие](exercises/production_ml_best_practices/task.md) · [папка](exercises/production_ml_best_practices/) · теория: [Production ML Best Practices](cheatsheets/production_ml_best_practices_cheatsheet.html)

**Этап 4: Специальные темы (2-3 недели)**

- **Дрейф данных: Population Stability Index** — [условие](exercises/concept_drift/task.md) · [папка](exercises/concept_drift/) · теория: [Дрейф данных и концептуальный дрейф](cheatsheets/concept_drift_cheatsheet.html)
- **Метрики справедливости и выравнивание TPR порогом** — [условие](exercises/fairness_ml/task.md) · [папка](exercises/fairness_ml/) · теория: [Fairness в ML](cheatsheets/fairness_ml_cheatsheet.html)
- **FGSM на логистической регрессии вручную и сертифицированная устойчивость** — [условие](exercises/adversarial_robustness/task.md) · [папка](exercises/adversarial_robustness/) · теория: [Adversarial robustness](cheatsheets/adversarial_robustness_cheatsheet.html)

### 📈 Путь 4: Специализация по доменам

**Computer Vision специалист**

- **IoU и Non-Maximum Suppression** — [условие](exercises/object_detection/task.md) · [папка](exercises/object_detection/) · теория: [Детекция объектов - Faster R-CNN, SSD, RetinaNet, EfficientDet](cheatsheets/object_detection_cheatsheet.html)
- **Метрики сегментации: IoU, Dice, pixel accuracy** — [условие](exercises/image_segmentation/task.md) · [папка](exercises/image_segmentation/) · теория: [Сегментация изображений (U-Net, DeepLab, Mask R-CNN)](cheatsheets/image_segmentation_cheatsheet.html)
- **Стереозрение: глубина по диспаритету и 3D-точки** — [условие](exercises/3d_computer_vision/task.md) · [папка](exercises/3d_computer_vision/) · теория: [3D Computer Vision](cheatsheets/3d_computer_vision_cheatsheet.html)
- **Style transfer: матрица Грама, content loss и style loss** — [условие](exercises/neural_style_transfer/task.md) · [папка](exercises/neural_style_transfer/) · теория: [Neural style transfer](cheatsheets/neural_style_transfer_cheatsheet.html)
- **Оценка позы: OKS и PCK по ключевым точкам COCO** — [условие](exercises/pose_estimation/task.md) · [папка](exercises/pose_estimation/) · теория: [Оценка позы человека](cheatsheets/pose_estimation_cheatsheet.html)

**NLP специалист**

- **TF-IDF вручную** — [условие](exercises/tfidf/task.md) · [папка](exercises/tfidf/) · теория: [TF-IDF и векторное пространство](cheatsheets/tfidf_cheatsheet.html)
- **Голова классификатора на CLS-векторах: warmup + linear decay** — [условие](exercises/bert_finetuning/task.md) · [папка](exercises/bert_finetuning/) · теория: [BERT fine-tuning](cheatsheets/bert_finetuning_cheatsheet.html)
- **NER: от BIO-тегов к сущностям и F1 по сущностям** — [условие](exercises/nlp_tasks_ner_classification_summarization/task.md) · [папка](exercises/nlp_tasks_ner_classification_summarization/) · теория: [Задачи NLP](cheatsheets/nlp_tasks_ner_classification_summarization_cheatsheet.html)
- **BPE вручную: список слияний на маленьком корпусе** — [условие](exercises/tokenization/task.md) · [папка](exercises/tokenization/) · теория: [Tokenization](cheatsheets/tokenization_cheatsheet.html)
- **NMF для тем: мультипликативные обновления Ли — Сына** — [условие](exercises/topic_modeling_lda_nmf/task.md) · [папка](exercises/topic_modeling_lda_nmf/) · теория: [Тематическое моделирование](cheatsheets/topic_modeling_lda_nmf_cheatsheet.html)

**Time Series специалист**

- **Экспоненциальное сглаживание: SES и Хольт** — [условие](exercises/exponential_smoothing/task.md) · [папка](exercises/exponential_smoothing/) · теория: [Экспоненциальное сглаживание](cheatsheets/exponential_smoothing_cheatsheet.html)
- 🐞 **Найди ошибку: валидация временного ряда** — [условие](exercises/bug_time_series_split/task.md) · [папка](exercises/bug_time_series_split/) · теория: [Валидация временных рядов](cheatsheets/time_series_validation_cheatsheet.html)
- **Лаговые признаки без утечки и walk-forward проверка** — [условие](exercises/time_series/task.md) · [папка](exercises/time_series/) · теория: [ARIMA / SARIMA](cheatsheets/time_series_cheatsheet.html)
- **Разложение «как Prophet»: тренд с изломом, Фурье и праздники МНК** — [условие](exercises/prophet_time_series/task.md) · [папка](exercises/prophet_time_series/) · теория: [Prophet](cheatsheets/prophet_time_series_cheatsheet.html)
- **Окна, скейлер по train и прогноз рекуррентной сетью** — [условие](exercises/rnn_lstm_time_series/task.md) · [папка](exercises/rnn_lstm_time_series/) · теория: [RNN/LSTM для временных рядов](cheatsheets/rnn_lstm_time_series_cheatsheet.html)
- **TCN: каузальная дилатированная свёртка и рецептивное поле** — [условие](exercises/cnn_time_series_tcn/task.md) · [папка](exercises/cnn_time_series_tcn/) · теория: [CNN для временных рядов (TCN)](cheatsheets/cnn_time_series_tcn_cheatsheet.html)
- **Патчи PatchTST и каузальное внимание по ним** — [условие](exercises/transformers_time_series/task.md) · [папка](exercises/transformers_time_series/) · теория: [Transformers для временных рядов](cheatsheets/transformers_time_series_cheatsheet.html)

**Reinforcement Learning специалист**

- **Q-learning и SARSA по записанной траектории** — [условие](exercises/q_learning_sarsa/task.md) · [папка](exercises/q_learning_sarsa/) · теория: [Q-learning и SARSA](cheatsheets/q_learning_sarsa_cheatsheet.html)
- **Многорукие бандиты: ε-greedy, UCB1 и Thompson на одном потоке наград** — [условие](exercises/multi_armed_bandits/task.md) · [папка](exercises/multi_armed_bandits/) · теория: [Multi-armed bandits](cheatsheets/multi_armed_bandits_cheatsheet.html)
- **Gridworld: оценка политики и итерация по ценности** — [условие](exercises/reinforcement_learning_basics/task.md) · [папка](exercises/reinforcement_learning_basics/) · теория: [RL Basics](cheatsheets/reinforcement_learning_basics_cheatsheet.html)
- **REINFORCE: возвраты, baseline и градиент softmax-политики** — [условие](exercises/policy_gradient/task.md) · [папка](exercises/policy_gradient/) · теория: [Policy Gradient методы](cheatsheets/policy_gradient_cheatsheet.html)
- **PPO: преимущества GAE и clipped-цель по шагам роллаута** — [условие](exercises/ppo_reinforcement_learning/task.md) · [папка](exercises/ppo_reinforcement_learning/) · теория: [PPO](cheatsheets/ppo_reinforcement_learning_cheatsheet.html)
- **DQN: мини-батч из replay buffer, цели target-сети и Double DQN** — [условие](exercises/dqn/task.md) · [папка](exercises/dqn/) · теория: [DQN](cheatsheets/dqn_cheatsheet.html)

### 🤖 Путь 5: ИИ-агенты (2-3 месяца)

**Этап 1: Основы агентов**

- **Симулятор цикла агента: шаги, стоимость и успешность при лимите шагов** — [условие](exercises/ai_agents_basics/task.md) · [папка](exercises/ai_agents_basics/) · теория: [Основы ИИ-агентов](cheatsheets/ai_agents_basics_cheatsheet.html)
- **Валидатор вызовов инструментов по JSON-схеме** — [условие](exercises/tool_use_function_calling/task.md) · [папка](exercises/tool_use_function_calling/) · теория: [Вызов инструментов (function calling)](cheatsheets/tool_use_function_calling_cheatsheet.html)
- **Разбор лога ReAct: повторы действий, зацикливание и эффективность** — [условие](exercises/react_agents/task.md) · [папка](exercises/react_agents/) · теория: [ReAct: рассуждение и действие](cheatsheets/react_agents_cheatsheet.html)

**Этап 2: Знания и память**

- **Векторный поиск: косинус, нормализация и IVF-подобное приближение** — [условие](exercises/vector_search/task.md) · [папка](exercises/vector_search/) · теория: [Векторный поиск (ANN, HNSW, FAISS)](cheatsheets/vector_search_cheatsheet.html)
- **RAG: чанкинг с перекрытием, recall@k и MRR** — [условие](exercises/rag/task.md) · [папка](exercises/rag/) · теория: [RAG: генерация с поиском](cheatsheets/rag_cheatsheet.html)
- **Память агента: буфер с вытеснением и подсчёт попаданий** — [условие](exercises/agent_memory/task.md) · [папка](exercises/agent_memory/) · теория: [Память агента](cheatsheets/agent_memory_cheatsheet.html)

**Этап 3: Планирование и команды агентов**

- **План агента: порядок подзадач, критический путь и бюджет токенов** — [условие](exercises/agent_planning/task.md) · [папка](exercises/agent_planning/) · теория: [Планирование и рефлексия агентов](cheatsheets/agent_planning_cheatsheet.html)
- **Стоимость координации: supervisor, все-со-всеми и конвейер** — [условие](exercises/multi_agent_systems/task.md) · [папка](exercises/multi_agent_systems/) · теория: [Мультиагентные системы](cheatsheets/multi_agent_systems_cheatsheet.html)
- **Разбор лога JSON-RPC: запросы, ответы, ошибки и таймауты** — [условие](exercises/mcp_protocol/task.md) · [папка](exercises/mcp_protocol/) · теория: [Model Context Protocol (MCP)](cheatsheets/mcp_protocol_cheatsheet.html)

**Этап 4: Надёжность и безопасность**

- **Оценка агента: несмещённые pass@k и pass^k по таблице попыток** — [условие](exercises/agent_evaluation/task.md) · [папка](exercises/agent_evaluation/) · теория: [Оценка агентов](cheatsheets/agent_evaluation_cheatsheet.html)
- **Безопасность агента: политика вызовов инструментов и «смертельная триада»** — [условие](exercises/agent_safety/task.md) · [папка](exercises/agent_safety/) · теория: [Безопасность агентов](cheatsheets/agent_safety_cheatsheet.html)

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
