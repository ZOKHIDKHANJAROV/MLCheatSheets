import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


df = pd.read_csv('input.csv')
X = np.column_stack([np.ones(len(df)), df[['x1', 'x2']].to_numpy()])   # столбец единиц — для свободного члена w0
y = df['y'].to_numpy()
n = len(y)

lr = 0.5
epochs = 30
w = np.zeros(3)
grad = np.zeros(3)

rows = []
for epoch in range(1, epochs + 1):
    p = sigmoid(X @ w)
    grad += X.T @ (p - y)
    w -= lr * grad
    loss = log_loss(y, sigmoid(X @ w))
    rows.append({'epoch': epoch, 'loss': loss, 'w0': w[0], 'w1': w[1], 'w2': w[2]})
    if epoch % 5 == 0:
        print(f'эпоха {epoch:2d}: loss = {loss:.4f}, w = {np.round(w, 3)}')

result = pd.DataFrame(rows).round(4)
result.to_csv('my_answer.csv', index=False)
print('Результат сохранён в my_answer.csv')
