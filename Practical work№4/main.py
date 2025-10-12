import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Загрузка данных из файла
df = pd.read_csv("youtube.csv")

#Для удобства удалим нечисловые колонки
nums_df = df.select_dtypes(include=['number'])
#Преобразование в словарь
dict = nums_df.to_dict(orient='list')
#Созранение в json
with open("numer_data_json", "w") as f:
    json.dump(dict, f, indent=4)
print("Данные сохранились в json")

#Предположение распределения и нормализация
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(nums_df)
scaler_df = pd.DataFrame(scaled_values, columns=nums_df.columns)
#Строим кореляционную матрицу
plt.figure(figsize=(8, 6))
sns.heatmap(scaler_df.corr(), annot = True, cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.show()

#Устранение выбросов
Q1 = scaler_df.quantile(0.25)
Q2 = scaler_df.quantile(0.75)
IQR = Q2 - Q1
filter_df = scaler_df[~((scaler_df < (Q1 - 1.5 * IQR)) | (scaler_df > (Q2 + 1.5* IQR))).any(axis=1)]
print(f"После удаления выбросов {filter_df.shape}")

#Удаляю дублирующиеся точки
filter_df = filter_df.drop_duplicates()
#Корреляционная матрица
plt.figure(figsize=(8, 6))
sns.heatmap(scaler_df.corr(), annot = True, cmap="coolwarm")
plt.title("Корреляционная матрица(Без выбрососв)")
plt.show()

#Преобразуем данные в числа
int_df = (filter_df * 100).astype(int)
#Уменьшение размерности
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(int_df)
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2', 'PC3'])

#Итоговая корреляционная матрица
plt.figure(figsize=(6,5))
sns.heatmap(reduced_df.corr(), annot=True, cmap="coolwarm")
plt.title("Корреляционная матрица (после PCA)")
plt.show()