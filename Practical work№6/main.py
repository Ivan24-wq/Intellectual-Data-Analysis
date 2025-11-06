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


#Определение признаков
exam_cols = ['math score', 'reading score', 'writing score']
X = df.drop(columns = exam_cols).values

#Выборка в пропорции 70/30
def train_test(X, y, test_ratio = 0.3, seed = 42):
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)  #Генератор случайных чисел
    test_size = int(len(X) * test_ratio)
    #Массив индесов
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


#Реализация линейное регресии
def linear_regresion(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta

#Предсказание
def predict(X, beta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ beta

#Оценка качества
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot


'''
Построение модели!
Экзамены независимы
'''
res_1 = {}
for exam in exam_cols:
    y = df[exam].values
    X_train, X_test, y_train, y_test = train_test(X, y)
    beta = linear_regresion(X_train, y_train)
    y_pred = predict(X_test, beta)
    res_1[exam] = r2_score(y_test, y_pred)

#Предположение: экзамены зависят
res_2 = {}   #Словарь резултатов
y = df['math score'].values
X_train, X_test, y_train, y_test = train_test(X, y)
beta_math = linear_regresion(X_train, y_train)
y_pred_math = predict(X_test, beta_math)
res_2['math score'] = r2_score(y_test, y_pred_math)


#Предсказание(математика)
X_read = np.c_[df['math score'].values, X]
y = df['reading score'].values
X_train, X_test, y_train, y_test = train_test(X_read, y)
beta_read = linear_regresion(X_train, y_train)
y_pred_read = predict(X_test, beta_read)
res_2['reading score'] = r2_score(y_test, y_pred_read)


X_write = np.c_[df[['math score', 'reading score']].values, X]
#Целевая переменная
y = df['writing score'].values
#Раздел данных
X_train, X_test, y_train, y_test = train_test(X_write, y)
#Обучение линейной регерссии
beta_write = linear_regresion(X_train, y_train)
y_pred_write = predict(X_test, beta_write)
res_2['writing score'] = r2_score(y_test, y_pred_write)

print("\n Квадрат результатов предположения 1")
for exam, r2 in res_1.items():
    print(f"{exam}: {r2:3f}")
    
print("\n Квадрат результатов предположения 2")
for exam, r2 in res_2.items():
    print(f"{exam}: {r2:3f}")