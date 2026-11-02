import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES = ['income', 'debt', 'age', 'credit_years']

df = pd.read_csv('input.csv')
X = df[FEATURES]
y = df['default'].to_numpy()

# Пропуски заполняем средним, признаки приводим к одному масштабу
X = SimpleImputer(strategy='mean').fit_transform(X)
X = StandardScaler().fit_transform(X)

train = (df['split'] == 'train').to_numpy()
test = ~train

model = LogisticRegression(C=0.1)
model.fit(X[train], y[train])
proba = model.predict_proba(X[test])[:, 1]
print(f'Обучение: {train.sum()} заявок, тест: {test.sum()} заявок')
print(f'ROC-AUC на тесте: {roc_auc_score(y[test], proba):.4f}')

result = pd.DataFrame({'id': df.loc[test, 'id'].to_numpy(), 'proba': proba.round(4)})
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
