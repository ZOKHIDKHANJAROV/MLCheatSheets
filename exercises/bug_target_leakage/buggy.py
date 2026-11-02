import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('input.csv')

# Берём все доступные колонки, кроме служебных и целевой
FEATURES = [c for c in df.columns if c not in ('id', 'default', 'split')]
print('Признаки:', FEATURES)

train = df[df['split'] == 'train']
test = df[df['split'] == 'test']

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(train[FEATURES], train['default'])
proba = model.predict_proba(test[FEATURES])[:, 1]
print(f'ROC-AUC на тесте: {roc_auc_score(test["default"], proba):.4f}')

result = pd.DataFrame({'id': test['id'].to_numpy(), 'proba': proba.round(4)})
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
