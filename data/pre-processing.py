import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf  

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
