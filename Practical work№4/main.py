import pandas as pd
import json
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