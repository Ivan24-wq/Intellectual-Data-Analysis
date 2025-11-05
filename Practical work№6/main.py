import pandas as pd
import numpy as np

#Путь к датасету
df = pd.read_csv("exams - exams.csv")
#Создаю числовой датасет
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]

print("Преобразованный датасет в числа")
print(df.head(), "\n")


#Список экзаменов
exam_cols = ['math score', 'reading score', 'writing store']
X = df.drop(columns = exam_cols).values

#Выборка в пропорции 70/30
def train_test(X, y, test_ratio = 0.3, seed = 42):
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def predict(X, beta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ beta

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot


#Построение модели
