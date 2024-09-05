import os
import pandas as pd
import numpy as np
import random

def load_and_preprocess_data(path, EMR_path):
    file_list = [file for file in os.listdir(path) if file.endswith('.xlsx')]
    dataFiles = [pd.read_excel(f"{path}/{file}") for file in file_list]

    all_data = []
    for data in dataFiles:
        data_0 = data[['타임스탬프(YYYY-MM-DDThh:mm:ss)', '이벤트 유형', '포도당 값 (mg/dL)', '인슐린 값(u)', '탄수화물 값 (그램)']]
        data_0 = data_0[data_0['이벤트 유형'].isin(["EGV", "인슐린", "탄수화물"])]
        data_0 = data_0.reset_index(drop=True)

        try:
            data_0['인슐린 값(u)'].iloc[data_0[data_0['인슐린 값(u)'].notnull()].index + 1] = \
            data_0['인슐린 값(u)'].iloc[data_0[data_0['인슐린 값(u)'].notnull()].index]
        except:
            pass

        data_0 = data_0[data_0['이벤트 유형'] == "EGV"].reset_index(drop=True)
        data_0 = data_0.replace(0.0, 1).fillna(0)
        all_data.append(data_0)

    EMR = pd.read_excel(EMR_path)
    EMR_10 = EMR.drop(['입원코드', '성별'], axis=1)

    for i in range(len(all_data)):
        all_data[i] = all_data[i].replace('높음', 400).replace('낮음', 40)
        all_data[i]['포도당 값 (mg/dL)'] = all_data[i]['포도당 값 (mg/dL)'].astype(float)
        all_data[i]['file_name'] = file_list[i]
        all_data[i] = pd.merge(all_data[i], EMR_10, how='left', on='file_name')
        all_data[i] = all_data[i].drop('file_name', axis=1)

    return all_data

def prepare_data(data):
    X_data, y_data, EMR_data = [], [], []
    for patient_data in data:
        X = patient_data[['포도당 값 (mg/dL)','인슐린 값(u)', '탄수화물 값 (그램)']]
        y = patient_data['포도당 값 (mg/dL)']
        EMR_x = patient_data[['나이', 'BMI', '당뇨 유병기간', 'BUN', 'Creatinine', 'CRP', 'C-peptide', 'HbA1c', 'Fructosamine', 'Urine Albumin/Creatinine ratio']]
        
        for j in range(len(X) - 13):
            X_data.append(X.iloc[j:j+7].values.tolist())
            EMR_data.append(EMR_x.iloc[0].values.tolist())
        
        for w in range(len(y) - 13):
            y_data.append(y.iloc[13 + w])
    
    return np.array(X_data), np.array(y_data), np.array(EMR_data)

# 클래스 레이블 생성
def create_class_labels(y_values):
    return [1 if y <= 70 else 2 if y >= 180 else 0 for y in y_values]
    
def split_train_test(data, train_size=81):
    # random.shuffle(data)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
