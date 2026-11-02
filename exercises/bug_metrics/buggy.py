import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_csv('input.csv')
y_true = df['y_true']
y_pred = (df['y_score'] >= 0.5).astype(int)

# Матрица ошибок для положительного класса (1)
tp = ((y_pred == 1) & (y_true == 1)).sum()
fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()
print(f'TP={tp}, FP={fp}, FN={fn}')

accuracy = (y_pred == y_true).mean()
precision = tp / (tp + fn)
recall = tp / (tp + fp)
f1 = f1_score(y_true, y_pred, average='macro')
roc_auc = roc_auc_score(y_true, y_pred)

result = pd.DataFrame([{
    'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
}]).round(4)
print(result.to_string(index=False))
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
