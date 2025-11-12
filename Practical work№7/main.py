import numpy as np
import pandas as pd



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
            