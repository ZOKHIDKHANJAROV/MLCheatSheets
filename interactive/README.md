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

## Как добавить демонстрацию

1. Создайте `interactive/<имя>.html` по образцу существующих: подключите `../assets/theme.js`, `lab.css` и `lab.js`.
2. Укажите связанные шпаргалки (имена без `_cheatsheet.html`):
   `<meta name="mlcs-topics" content="linreg regression_metrics">`
3. Заголовок страницы — `<title>Название — интерактив</title>`: часть до « — » станет названием в портале.
4. Пересоберите портал: `python tools/build_portal.py` (или просто `python start.py`).

`lab.js` даёт помощники: `Lab.canvas` (холст под размер экрана), `Lab.color` (цвета текущей темы), `Lab.slider`,
`Lab.rng` (воспроизводимые случайные числа), `Lab.frame` и `Lab.axes` (шкалы и оси).
