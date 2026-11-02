# 🎮 Интерактивные демонстрации

Живые визуализации к шпаргалкам: ползунки, клики, пошаговое выполнение алгоритмов. Работают офлайн (без библиотек),
поддерживают тёмную тему и открываются в портале кнопкой **🎮 Интерактив** у связанных тем.

| Демонстрация | Темы |
|---|---|
| [gradient_descent.html](gradient_descent.html) — спуск по поверхности потерь, learning rate, momentum, расходимость | градиентный спуск, оптимизаторы |
| [linreg.html](linreg.html) — ручной подбор прямой против МНК, квадраты ошибок, выбросы | линейная регрессия, метрики регрессии, регуляризация |
| [overfitting.html](overfitting.html) — степень полинома, U-кривая ошибок, Ridge | переобучение, bias–variance, полиномиальная регрессия |
| [knn.html](knn.html) — карта решений, соседи под курсором, leave-one-out от k | k-NN |
| [roc_threshold.html](roc_threshold.html) — порог, матрица ошибок, precision/recall, ROC-кривая | ROC-AUC, матрица ошибок, метрики классификации |
| [kmeans.html](kmeans.html) — шаги назначения и обновления, инициализация, метод локтя | k-means, выбор числа кластеров |
| [decision_tree.html](decision_tree.html) — разбиения CART на плоскости, правила дерева, accuracy от глубины | деревья решений |
| [gradient_boosting.html](gradient_boosting.html) — деревья по одному на остатках, learning rate, ранняя остановка | бустинг, XGBoost, LightGBM, CatBoost |
| [svm.html](svm.html) — зазор, опорные векторы, C, ядра linear/RBF/poly | SVM |
| [perceptron.html](perceptron.html) — правило обучения по шагам, сходимость и XOR | перцептрон |
| [mlp_playground.html](mlp_playground.html) — песочница MLP: признаки, слои, активации, backprop, L2, карты нейронов | MLP, backpropagation, затухание градиента |
| [activation_functions.html](activation_functions.html) — f(x) и f′(x), затухание и взрыв градиента, инициализация He | функции активации |
| [convolution.html](convolution.html) — ядро 3×3 по шагам, stride, padding, ReLU, max pooling, рисование мышью | основы CNN, пулинг |
| [pca.html](pca.html) — главные компоненты облака точек, дисперсия вдоль любого направления, стандартизация | PCA, снижение размерности |
| [rl_gridworld.html](rl_gridworld.html) — Q-learning против SARSA на «обрыве», карта ценностей, value iteration | основы RL, Q-learning, SARSA, DP |
| [attention.html](attention.html) — тепловая карта softmax(QKᵀ/√d), масштабирование, каузальная маска, головы, позиции | self-attention, трансформеры |
| [dropout_batchnorm.html](dropout_batchnorm.html) — переобучение с dropout и без; активации по слоям с BatchNorm и без | dropout, BatchNorm, инициализация |
| [embeddings.html](embeddings.html) — word2vec (skip-gram + negative sampling) учится в браузере: слова собираются в темы, окно, негативы, PCA-проекция, соседи по косинусу | эмбеддинги слов |
| [regularization_l1_l2.html](regularization_l1_l2.html) — ромб L1 против круга L2 и Elastic Net, касание линий уровня, путь коэффициентов Lasso/Ridge, обнуление и коррелированные признаки | регуляризация, Ridge/Lasso/ElasticNet, отбор признаков |
| [cross_validation.html](cross_validation.html) — схемы разбиения (hold-out, K-Fold, Stratified, Group, TimeSeriesSplit, LOO), оценка k-NN против истинного качества, разброс по seed, утечка при дрейфе и группах | кросс-валидация, валидация временных рядов, несбалансированные данные |
| [exponential_smoothing.html](exponential_smoothing.html) — скользящее среднее, SES, Холт и Холт–Винтерс на ряде с трендом и сезонностью, прогноз и MAE на тесте, подбор α/β/γ, веса α(1−α)^k | экспоненциальное сглаживание, скользящие средние, временные ряды |
| [gan.html](gan.html) — 1D GAN: плотности и D(x), отображение z → G(z), mode collapse на смеси гауссиан, non-saturating против minimax | GAN, DCGAN и Conditional GAN |
| [vae.html](vae.html) — латент VAE на картинках 12×12: атлас декодера, зонд и интерполяция, β-VAE, posterior collapse, сравнение с автоэнкодером | VAE, автоэнкодеры, вариационный вывод |

## Как добавить демонстрацию

1. Создайте `interactive/<имя>.html` по образцу существующих: подключите `../assets/theme.js`, `lab.css` и `lab.js`.
2. Укажите связанные шпаргалки (имена без `_cheatsheet.html`):
   `<meta name="mlcs-topics" content="linreg regression_metrics">`
3. Заголовок страницы — `<title>Название — интерактив</title>`: часть до « — » станет названием в портале.
4. Пересоберите портал: `python tools/build_portal.py` (или просто `python start.py`).

`lab.js` даёт помощники: `Lab.canvas` (холст под размер экрана), `Lab.color` (цвета текущей темы), `Lab.slider`,
`Lab.rng` (воспроизводимые случайные числа), `Lab.frame` и `Lab.axes` (шкалы и оси).
