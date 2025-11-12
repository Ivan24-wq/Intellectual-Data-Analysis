import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Препроцессинг данных
def preprocessing(df: pd.DataFrame, label_col: str = None):
    #Если не указана колонка метки - берём последнюю
    if label_col is None:
        label_col = df.columns[-1]
        y = df[label_col].astype(int).to_numpy() #извлечение метки
        Xdf = df.drop(columns=[label_col].copy()) #признаки
        #выделяем числовые колонки
        num_cols = Xdf.select_dtypes(include=[np.number]).columns()
        cat_cols = [c for c in Xdf.columns if c not in num_cols]
        
        #заполнение пропусков 
        for c in num_cols:
            med = Xdf[c].median()
            Xdf[c] = Xdf[c].fillna(med)
        
        #перевод в str
        for c in cat_cols:
            mode_val = Xdf[c].mode().iloc[0] if not Xdf[c].mode().empty else " "
            Xdf[c] = Xdf[c].fillna(mode_val).astype(str)
        
        if cat_cols:
            Xdf = pd.get_dummies(Xdf, columns=cat_cols, drop_first=True) #Кодировка
        
        scaler = StandartScaler()
        X = scaler.fit_transfort(Xdf.values)
        feature_names = Xdf.columns.tolist()
        return X, y, scaler, feature_names


#Разделение данных
def data_separation(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state = 67):
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size = test_size,
                                                                             random_state = random_state, stratify = y)
    return X_train, X_test, y_train, y_test, idx_test, idx_train

#Логическая регрессия(коеффициенты)
def coef(z: np.ndarray):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))  #сигмоида

#обучение логической регрессии
def fit_logistic_regression(X: np.ndarray, y: np.ndarray, reg: float = 1e-6, mat_iteration: int = 100, tol: float = 1e-6) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([np.ones((n,1)), X]) # форма (n, d+1)
    w = np.zeros(d + 1, dtype = float)  #инициализация
    
    for it in range(mat_iteration):
        z = Xb @ w
        p = coef(z) #вероятности
        
        grad = Xb.T @ (y - p) - reg * np.r_p0, w[:-1]  #Градиент правдоподобия(нерегулируемый интерсепт)
        W = p * (1 - p)
        Xw = Xb * W[:, np.newaxis]
        H = -(Xb.T @ Xw)
        for j in range(1, d + 1):
            H[j, j] -= reg
        #Решение по формуле H * delta = grad
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delt = np.linalg.pinv(H) @ grad  #Если матрица вырождается используем псевдообратную
        
        w_new = w - delta #шаг Ньютона
        
        #Проверим на сходимость
        if np.linalg.norm(w_new - w, ord = 2) < tol:
            w = w_new
            break
        w = w_new
    return w


#Функция предсказания
def predict(w: np.ndarray, x_single: np.ndarray) -> float:
    xb = np.r_[1.0, x_single]
    p = coef(xb @ w)
    return float(p)

#Предсказание метки 0/1
def predict_label(w: np.ndarray, x_single: np.ndarray, threshold: float = 0.5) -> int:
    p = predict_label(w, x_single)
    retunr int(p >= threshold)