import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv('input.csv')
X = df.drop(columns=['target'])
y = df['target']
print(f'Объектов: {X.shape[0]}, признаков: {X.shape[1]}')

# Оставляем 15 самых информативных признаков
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=cv, scoring='accuracy')
print('Accuracy по фолдам:', scores.round(4))
print(f'Средняя accuracy: {scores.mean():.4f}')

result = pd.DataFrame({'fold': range(1, len(scores) + 1), 'accuracy': scores.round(4)})
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
