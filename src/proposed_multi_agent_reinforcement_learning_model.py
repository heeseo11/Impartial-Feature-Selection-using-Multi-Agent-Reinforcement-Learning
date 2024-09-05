import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Input, Flatten, Concatenate, RepeatVector
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
import random

# Random seed 설정
tf.random.set_seed(42)
random.seed(42)

# 데이터 불러오기 및 전처리
path = '/Users/seohee/Documents/BloodGlucose/Data/glucose data'
file_list = [file for file in os.listdir(path) if file.endswith('.xlsx')]
dataFiles = [pd.read_excel(f"{path}/{file}") for file in file_list]

all_data = []
for i, data in enumerate(dataFiles):
    data_0 = data[['타임스탬프(YYYY-MM-DDThh:mm:ss)', '이벤트 유형', '포도당 값 (mg/dL)', '인슐린 값(u)', '탄수화물 값 (그램)']]
    data_0 = data_0[data_0['이벤트 유형'].isin(["EGV", "인슐린", "탄수화물"])]
    data_0 = data_0.reset_index(drop=True)

    # 인슐린 값 이동 및 전처리
    try:
        data_0['인슐린 값(u)'].iloc[data_0[data_0['인슐린 값(u)'].notnull()].index + 1] = \
        data_0['인슐린 값(u)'].iloc[data_0[data_0['인슐린 값(u)'].notnull()].index]
    except:
        pass

    data_0 = data_0[data_0['이벤트 유형'] == "EGV"].reset_index(drop=True)
    try:
        data_0['탄수화물 값 (그램)'].iloc[data_0[data_0['탄수화물 값 (그램)'].notnull()].index + 1] = \
        data_0['탄수화물 값 (그램)'].iloc[data_0[data_0['탄수화물 값 (그램)'].notnull()].index]
    except:
        pass

    data_0 = data_0.replace(0.0, 1).fillna(0)
    in_tan = list(set(data_0[data_0['인슐린 값(u)'] != 0.0].index) & set(data_0[data_0['탄수화물 값 (그램)'] != 0.0].index))

    for idx in set(in_tan) & set(data_0[data_0['이벤트 유형'] == '탄수화물'].index):
        data_0['인슐린 값(u)'].iloc[idx + 1] = data_0['인슐린 값(u)'].iloc[idx]

    data_0 = data_0[data_0['이벤트 유형'] == "EGV"]
    all_data.append(data_0)

EMR = pd.read_excel("/Users/seohee/Documents/BloodGlucose/Data/EMR.xlsx")
EMR_10 = EMR.drop(['입원코드', '성별'], axis=1)

# 데이터 병합 및 후처리
for i in range(len(all_data)):
    all_data[i] = all_data[i].replace('높음', 400).replace('낮음', 40)
    all_data[i]['포도당 값 (mg/dL)'] = all_data[i]['포도당 값 (mg/dL)'].astype(float)
    all_data[i].loc[all_data[i]['인슐린 값(u)'] != 0.0, '인슐린 값(u)'] = 1
    all_data[i].loc[all_data[i]['탄수화물 값 (그램)'] != 0.0, '탄수화물 값 (그램)'] = 1
    all_data[i]['file_name'] = file_list[i]
    all_data[i] = pd.merge(all_data[i], EMR_10, how='left', on='file_name')
    all_data[i] = all_data[i].drop('file_name', axis=1)

random.shuffle(all_data)  # 환자별 데이터 랜덤 셔플

# Train/Test 데이터 준비
train_data = all_data[:81]
test_data = all_data[81:]

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

train_X, train_y, train_EMR = prepare_data(train_data)
test_X, test_y, test_EMR = prepare_data(test_data)

# 클래스 레이블 생성
def create_class_labels(y_values):
    return [1 if y <= 70 else 2 if y >= 180 else 0 for y in y_values]

train_class = create_class_labels(train_y)
test_class = create_class_labels(test_y)

# NumPy 배열로 변환
train_y = pd.DataFrame({'포도당 값 (mg/dL)': train_y, 'class': train_class}).values
test_y = pd.DataFrame({'포도당 값 (mg/dL)': test_y, 'class': test_class}).values

train_y_pred = train_y[:, 0]
train_y_class = train_y[:, 1]
test_y_pred = test_y[:, 0]
test_y_class = test_y[:, 1]

# 원-핫 인코딩
y_train_class = to_categorical(train_y_class.astype('float32'))
y_test_class = to_categorical(test_y_class.astype('float32'))

# 모델 정의
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class T2V(tf.keras.layers.Layer):
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.output_dim), initializer='uniform', trainable=True)
        self.P = self.add_weight(name='P', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        self.p = self.add_weight(name='p', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        original = self.w * x + self.p
        sin_trans = tf.sin(tf.linalg.matmul(x, self.W) + self.P)
        return tf.concat([sin_trans, original], -1)

def attention_model(EMR_N):
    n_hidden = 20
    
    input_train = Input(shape=(7, 3))
    output_train = Input((1,))
    EMR_input = Input(shape=(EMR_N,))
    
    x = T2V(16)(input_train)
    encoder_last_h, encoder_last_c = GRU(n_hidden, activation='tanh', return_state=True, return_sequences=False)(x)
    encoder_stack_h = GRU(n_hidden, activation='tanh', return_sequences=True)(x)

    encoder_last_h = BatchNormalization()(encoder_last_h)
    encoder_last_c = BatchNormalization()(encoder_last_c)

    decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
    decoder_stack_h = GRU(n_hidden, activation='tanh', return_sequences=True)(decoder_input)

    attention = BahdanauAttention(8)
    context_vector, attention_weights = attention(encoder_stack_h, decoder_stack_h)
    context = BatchNormalization()(context_vector)

    decoder_combined_context = tf.keras.layers.dot([decoder_stack_h, context], axes=[2, 2])
    hidden = Flatten()(decoder_combined_context)

    concatted = Concatenate()([EMR_input, hidden])
    out_clas = Dense(3, activation='softmax')(concatted)

    model2 = Model(inputs=[input_train, EMR_input], outputs=out_clas)
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model2

# Accuracy 함수 정의
def accuracy(input):
    X_train_EMR = tf.convert_to_tensor(train_EMR[:, input])
    X_test_EMR = tf.convert_to_tensor(test_EMR[:, input])
    
    clf = attention_model(len(X_train_EMR[0]))
    
    Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    clf.fit([train_X, X_train_EMR], y_train_class, validation_split=0.2, epochs=50, batch_size=20, verbose=1, callbacks=[Early_stop])
    
    pred = clf.predict([test_X, X_test_EMR])
    pred_arg = np.argmax(pred, axis=1)
    
    y_test_class = np.argmax(test_y, axis=1)
    
    f1 = f1_score(y_test_class, pred_arg, average='macro')
    return f1

# Q-learning 관련 코드
Q_values = [[-1,-1]] * 10

def get_reward(features):
    if not features:
        return 0
    acc = accuracy(features) * 100
    tot_f = len(features)
    R = acc
    if tot_f > K_len:
        R = acc * K_len / tot_f
    return R

epsilon = 0.4
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995
K_len = 5

all_rewards = []
num_episodes = 50
num_agents = 10
reward_store = {i: 0 for i in range(num_agents)}
reward_store_array = []
agent_reward_store_array = []

#actions = [0] * num_agents
with tf.device('/CPU:0'):
    for episode in range(num_episodes):
        print("num_episodes:", episode)
        
        # 각 에이전트의 행동 선택
        actions = [
            np.argmax(Q_values[agent]) if random.uniform(0,1) > epsilon else random.choice([0,1]) 
            for agent in range(num_agents)
        ]

        total_features = [i for i, act in enumerate(actions) if act == 1]
        R = get_reward(total_features)
        reward_store[len(total_features)-1] = max(reward_store[len(total_features)-1], R)
        reward_store_array.append(["R", total_features, R])
        all_rewards.append(R)

        # 각 에이전트의 개별 행동 업데이트
        for agent in range(num_agents):
            individual_action = actions[:]
            individual_action[agent] = 1 - individual_action[agent]  # 반전된 행동

            individual_features = [i for i, act in enumerate(individual_action) if act == 1]

            RI = get_reward(individual_features)
            plus_reward = R / len(total_features) if actions[agent] == 1 else 0
            C_agent = R - RI + plus_reward if actions[agent] == 1 else R - RI

            agent_reward_store_array.append(["A_R", individual_features, R, C_agent])
            Q_values[agent][actions[agent]] += alpha * (C_agent - Q_values[agent][actions[agent]])

        # 탐색-활용 균형 감소
        alpha *= alpha_decay_rate
        epsilon *= epsilon_decay_rate
        print("Q_values:", Q_values)
