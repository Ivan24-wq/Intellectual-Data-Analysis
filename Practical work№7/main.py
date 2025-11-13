import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Tuple 

# Препроцессинг данных
def preprocessing(df: pd.DataFrame, label_col: str = None):
    # Если не указана колонка метки - берём последнюю
    if label_col is None:
        label_col = df.columns[-1]
    y = df[label_col].astype(int).to_numpy()  # извлечение метки
    Xdf = df.drop(columns=[label_col]).copy()  # признаки

    # выделяем числовые колонки
    num_cols = Xdf.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in Xdf.columns if c not in num_cols]
    
    # заполнение пропусков 
    for c in num_cols:
        med = Xdf[c].median()
        Xdf[c] = Xdf[c].fillna(med)
    
    # перевод категориальных в строку и заполнение модой
    for c in cat_cols:
        mode_val = Xdf[c].mode().iloc[0] if not Xdf[c].mode().empty else " "
        Xdf[c] = Xdf[c].fillna(mode_val).astype(str)
    
    # кодировка категориальных признаков
    if cat_cols:
        Xdf = pd.get_dummies(Xdf, columns=cat_cols, drop_first=True)
    
    # стандартизация
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf.values)
    feature_names = Xdf.columns.tolist()
    return X, y, scaler, feature_names


# Разделение данных
def data_separation(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 67):
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


# Логическая регрессия (сигмоида)
def coef(z: np.ndarray):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))  # сигмоида


# обучение логической регрессии
def fit_logistic_regression(X: np.ndarray, y: np.ndarray, reg: float = 1e-6, mat_iteration: int = 100, tol: float = 1e-6) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])  # форма (n, d+1)
    w = np.zeros(d + 1, dtype=float)  # инициализация
    
    for it in range(mat_iteration):
        z = Xb @ w
        p = coef(z)  # вероятности

        grad = Xb.T @ (y - p)
        grad[1:] -= reg * w[1:]  # регуляризация без интерсепта
        
        W = p * (1 - p)
        Xw = Xb * W[:, np.newaxis]
        H = -(Xb.T @ Xw)
        for j in range(1, d + 1):
            H[j, j] -= reg

        # Решение по формуле H * delta = grad
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad  # Если матрица вырождается, используем псевдообратную
        
        w_new = w - delta  # шаг Ньютона
        
        # Проверим на сходимость
        if np.linalg.norm(w_new - w, ord=2) < tol:
            w = w_new
            break
        w = w_new
    return w


# Функция предсказания
def predict(w: np.ndarray, x_single: np.ndarray) -> float:
    xb = np.r_[1.0, x_single]
    p = coef(xb @ w)
    return float(p)


# Предсказание метки 0/1
def predict_label(w: np.ndarray, x_single: np.ndarray, threshold: float = 0.5) -> int:
    p = predict(w, x_single)
    return int(p >= threshold)


# Выбор нестабильных экземпляров (вероятность около 0.5)
def selected_regression(probs: np.ndarray, target_fraction: float = 0.2):
    # Нахождение минимального порога
    diffs = np.abs(probs - 0.5)
    sorted_diffs = np.sort(diffs)
    N = len(diffs)
    k = max(1, int(np.ceil(target_fraction * N)))
    delta = sorted_diffs[k - 1]
    unstable_mask = diffs <= (delta + 1e-12)
    return unstable_mask, delta


# Построение классификации (метод ближайших соседей)
def classification(X_train: np.ndarray, y_train: np.ndarray, k: int = 5) -> KNeighborsClassifier:
    # Обучает ИИ
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


# Метрики качества
def metric(y_true: np.ndarray, y_pred: np.ndarray):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1


def compute(y_true_ter: np.ndarray, y_pred_ter: np.ndarray):
    return accuracy_score(y_true_ter, y_pred_ter)


# Запуск скрипта
def start(csv_path: str, label_col: str = None, test_size: float = 0.3, random_state: int = 42, k_knn: int = 5):
    df = pd.read_csv(csv_path)
    # Препроцессинг
    X_all, y_all, scaler, feature_names = preprocessing(df, label_col=label_col)
    # Деление данных
    X_train, X_test, y_train, y_test, idx_train, idx_test = data_separation(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    # Логистическая регрессия
    w = fit_logistic_regression(X_train, y_train, reg=1e-4, mat_iteration=150)
    n_all = X_all.shape[0]
    Xb_all = np.hstack([np.ones((n_all, 1)), X_all])
    probs_all = coef(Xb_all @ w)
    
    # Выборка нестабильных экземпляров
    unstable_mask_all, delta = selected_regression(probs_all, target_fraction=0.2)
    unstable_fraction = unstable_mask_all.mean()

    # Метод ближайших соседей
    knn = classification(X_train, y_train, k=k_knn)
    
    # Предсказания на тесте
    n_test = X_test.shape[0]
    Xb_test = np.hstack([np.ones((n_test, 1)), X_test])
    probs_test = coef(Xb_test @ w)
    y_pred_lr_test = (probs_test >= 0.5).astype(int)
    y_pred_knn_test = knn.predict(X_test)

    # Маска нестабильных объектов ТОЛЬКО для тестовой выборки
    unstable_test_mask = unstable_mask_all[idx_test]  # теперь длина = длине y_test

    # Преобразуем в тернарные метки (0/1/2)
    y_true_ter = y_test.copy()
    y_pred_lr_ter = y_pred_lr_test.copy()
    y_pred_knn_ter = y_pred_knn_test.copy()

    # применяем флаг нестабильности для теста
    for i in range(len(y_test)):
        if unstable_test_mask[i]:
            y_true_ter[i] = 2
            y_pred_lr_ter[i] = 2
            y_pred_knn_ter[i] = 2

    # Бинарные метрики (только стабильная часть)
    stable_mask_test = np.logical_not(unstable_test_mask)  # маска длиной len(y_test)
    num_stable = stable_mask_test.sum()
    if num_stable > 0:
        p_lr, r_lr, f1_lr = metric(y_test[stable_mask_test], y_pred_lr_test[stable_mask_test])
        p_knn, r_knn, f1_knn = metric(y_test[stable_mask_test], y_pred_knn_test[stable_mask_test])
    else:
        p_lr = r_lr = f1_lr = np.nan
        p_knn = r_knn = f1_knn = np.nan
    
    # Точность для тернарной классификации
    acc_lr_ter = compute(y_true_ter, y_pred_lr_ter)
    acc_knn_ter = compute(y_true_ter, y_pred_knn_ter)
      
    print("=== Результаты ===")
    print(f"Всего объектов: {len(y_all)}")
    print(f"Доля нестабильных (всего датасет): {unstable_fraction:.3f} (требуется ≥0.20)")
    print(f"delta для критерия нестабильности: {delta:.6f}")
    print(f"Размер теста: {n_test}, стабильных в тесте: {num_stable} ({num_stable/n_test:.3f})")
    print("\n-- Бинарные метрики (только на стабильной части теста) --")
    print(f"LR: precision={p_lr:.4f}, recall={r_lr:.4f}, f1={f1_lr:.4f}")
    print(f"kNN(k={k_knn}): precision={p_knn:.4f}, recall={r_knn:.4f}, f1={f1_knn:.4f}")
    print("\n-- Точность для тернарной классификации (Accuracy on test, labels 0/1/2) --")
    print(f"LR accuracy (ternary) = {acc_lr_ter:.4f}")
    print(f"kNN accuracy (ternary) = {acc_knn_ter:.4f}")


if __name__ == "__main__":
    csv_path = "/Users/mac/Documents/IAD/Intellectual-Data-Analysis/Practical work№7/heart.csv"
    start(csv_path, label_col=None, test_size=0.3, random_state=42, k_knn=5)
