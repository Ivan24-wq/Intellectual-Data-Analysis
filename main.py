import pandas as pd
import matplotlib.pyplot as plt

def main(file):
    #Загрузка данных
    df = pd.read_csv(file, header=None, names=['date', 'temperature'], encoding='cp1251', sep=';', usecols=[0, 1])
    
    #Обработка температур
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')#Преобразуем в числа если можем
    df = df.dropna(subset=['temperature']) #Удаляем всё что не преобразовалось(мусор если он есть)
    
    #Обработка дат
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M', errors='coerce')#Преобразуем числа в дату
    df = df.dropna(subset=['date']) #Удаляем всё что не преобразовалось(мусор если он есть)
    
    #Среднесуточные температуры
    df['date_only'] = df['date'].dt.date # Добавляем колонку дата БЕЗ ВРЕМЕНИ(в таблице есть несколько замеров в один день но в разное время)
    daily_avg = df.groupby('date_only')['temperature'].mean().reset_index() #Считаем средние температуры в день
    
    
    #Добавляем месяц
    daily_avg['month'] = pd.to_datetime(daily_avg['date_only']).dt.month #Добовляем колонку месяц 1-12
    
    #Стандартное отклонение по месяцам
    monthly_std = daily_avg.groupby('month')['temperature'].std().reset_index()#Стандартное отклонение по месяцам
    monthly_std.columns = ['month', 'std_deviation'] #Для удобства оставляем только колонку месяц и отколонение
    
    return monthly_std

msk = main(r'F:\ИАД\Intellectual-Data-Analysis\moskov.csv')
anadyr = main(r'F:\ИАД\Intellectual-Data-Analysis\anadyr.csv')

#Построение графика
plt.figure(figsize=(12, 6))
plt.plot(msk['month'], msk['std_deviation'], marker='o', label='Москва', linewidth=2, markersize=8)
plt.plot(anadyr['month'], anadyr['std_deviation'], marker='s', label='Анадырь', linewidth=2, markersize=8)

#Настройки графика
plt.title('Стандартное отклонение температур по месяцам', fontsize=14)
plt.xlabel('Месяц', fontsize=12)
plt.ylabel('Стандартное отклонение (°C)', fontsize=12)
plt.xticks(range(1, 13), ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
