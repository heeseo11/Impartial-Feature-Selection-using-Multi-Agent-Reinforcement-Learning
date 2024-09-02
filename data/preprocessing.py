import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf  
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
# tf.random.set_seed(42)
random.seed(42)

EMR = pd.read_excel("./EMR.xlsx")
EMR_10 = EMR.drop(['입원코드', '성별'], axis = 1)

# Define the path to the data directory
path = './glucose data'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.xlsx')]

# Load Excel files into a list of DataFrames
dataFiles = [pd.read_excel(os.path.join(path, file)) for file in file_list_py]

# Process each DataFrame
all_data = []
for i in range(len(dataFiles)):
    
    data_0 = dataFiles[i][['타임스탬프(YYYY-MM-DDThh:mm:ss)', '이벤트 유형', '포도당 값 (mg/dL)', '인슐린 값(u)', '탄수화물 값 (그램)']]
    data_0 = data_0[(data_0['이벤트 유형'] == "EGV") | 
                    (data_0['이벤트 유형'] == "인슐린") | 
                    (data_0['이벤트 유형'] == "탄수화물")]
    
    data_0 = data_0.reset_index(drop=True)

    # Shift insulin values forward by one row
    try:
        data_0['인슐린 값(u)'][data_0[data_0['인슐린 값(u)'].notnull()].index + 1] = \
        data_0['인슐린 값(u)'][data_0[data_0['인슐린 값(u)'].notnull()].index]
    except:
        pass

    # Remove insulin rows
    data_0 = data_0[(data_0['이벤트 유형'] == "EGV") | (data_0['이벤트 유형'] == "탄수화물")]
    data_0 = data_0.reset_index(drop=True)

    # Shift carbohydrate values forward by one row
    try:
        data_0['탄수화물 값 (그램)'][data_0[data_0['탄수화물 값 (그램)'].notnull()].index + 1] = \
        data_0['탄수화물 값 (그램)'][data_0[data_0['탄수화물 값 (그램)'].notnull()].index]
    except:
        pass

    # Replace 0.0 with 1 and NaN with 0
    data_0 = data_0.replace(0.0, 1).replace(np.nan, 0)

    # Adjust insulin and carbohydrate values
    in_tan = list(set(data_0[data_0['인슐린 값(u)'] != 0.0].index)
                  .intersection(data_0[data_0['탄수화물 값 (그램)'] != 0.0].index))
    
    for j in list((set(in_tan)) & set(list(data_0[data_0['이벤트 유형'] == '탄수화물'].index))):
        data_0['인슐린 값(u)'][j + 1] = data_0['인슐린 값(u)'][j]

    # Keep only EGV rows
    data_0 = data_0[data_0['이벤트 유형'] == "EGV"]

    # Print processed data summary
    # print(sum(data_0['탄수화물 값 (그램)'] != 0.0))
    # print(sum(data_0['인슐린 값(u)'] != 0.0))
    # print("----------------------------------다음 환자 이동 ---------------------------------------")
    
    all_data.append(data_0)

# Post-processing all data
for i in range(len(all_data)):
    all_data[i] = all_data[i].replace('높음', 400)
    all_data[i] = all_data[i].replace('낮음', 40)
    all_data[i].loc[all_data[i]['인슐린 값(u)'] != 0.0, '인슐린 값(u)'] = 1
    all_data[i].loc[all_data[i]['탄수화물 값 (그램)'] != 0.0, '탄수화물 값 (그램)'] = 1
    all_data[i] = all_data[i][['포도당 값 (mg/dL)', '인슐린 값(u)', '탄수화물 값 (그램)']]
    all_data[i] = all_data[i].reset_index(drop=True)

# Merging with additional data and final adjustments
for i in range(len(all_data)):
    all_data[i]['file_name'] = file_list_py[i]
    all_data[i] = pd.merge(all_data[i], EMR_10, how='left', on='file_name')
    all_data[i] = all_data[i].drop('file_name', axis=1)

    arr = np.array(all_data[i]['포도당 값 (mg/dL)'])
    arr = np.where(arr == "높음", 400, arr)
    arr = np.where(arr == "낮음", 40, arr)
    all_data[i]['포도당 값 (mg/dL)'] = arr

# 데이터 셔플 및 train/test 분리
random.shuffle(all_data)  # 환자별 데이터 랜덤 셔플

train_data = all_data[:81]
test_data = all_data[81:]

# 데이터 준비 함수 정의
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

# Train/Test 데이터 준비
train_X, train_y, train_EMR = prepare_data(train_data)
test_X, test_y, test_EMR = prepare_data(test_data)

# 클래스 레이블 생성 함수 정의
def create_class_labels(y_values):
    return [1 if y <= 70 else 2 if y >= 180 else 0 for y in y_values]

train_class = create_class_labels(train_y)
test_class = create_class_labels(test_y)

# 데이터프레임으로 변환
train_y_df = pd.DataFrame({'포도당 값 (mg/dL)': train_y, 'class': train_class})
test_y_df = pd.DataFrame({'포도당 값 (mg/dL)': test_y, 'class': test_class})

# NumPy 배열로 변환
train_y = train_y_df.values
test_y = test_y_df.values

# 각 변수 분리
train_y_pred = train_y[:, 0]
train_y_class = train_y[:, 1]
test_y_pred = test_y[:, 0]
test_y_class = test_y[:, 1]

# 원-핫 인코딩
y_train_class = to_categorical(train_y_class.astype('float32'))
y_test_class = to_categorical(test_y_class.astype('float32'))

# 데이터셋 모양 출력
print('train_X shape:', train_X.shape)
print('test_X shape:', test_X.shape)
print('train_y_pred shape:', train_y_pred.shape)
print('train_y_class shape:', train_y_class.shape)
print('test_y shape:', test_y.shape)
print('train_EMR shape:', train_EMR.shape)
print('test_EMR shape:', test_EMR.shape)

# EMR 데이터를 데이터프레임으로 변환
train_EMR_df = pd.DataFrame(train_EMR, columns=['나이', 'BMI', '당뇨 유병기간', 'BUN', 'Creatinine', 'CRP', 'C-peptide', 'HbA1c', 'Fructosamine', 'Urine Albumin/Creatinine ratio'])
test_EMR_df = pd.DataFrame(test_EMR, columns=['나이', 'BMI', '당뇨 유병기간', 'BUN', 'Creatinine', 'CRP', 'C-peptide', 'HbA1c', 'Fructosamine', 'Urine Albumin/Creatinine ratio'])

