from data_preprocessing import load_and_preprocess_data, split_train_test, prepare_data, create_class_labels
from reinforcement_learning import q_learning
from model import attention_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Data Loading and Preprocessing
data_path = './glucose data'
EMR_path = './EMR.xlsx'
all_data = load_and_preprocess_data(data_path, EMR_path)

# Train/Test Data Split
train_data, test_data = split_train_test(all_data)

# Return train_EMR, test_EMR, train_X, test_X, y_train_class, and y_test_class in the prepare_data function
train_X, train_y, train_EMR = prepare_data(train_data)
test_X, test_y, test_EMR = prepare_data(test_data)

# Creating Class Labels
train_class = create_class_labels(train_y)
test_class = create_class_labels(test_y)

# Convert List to NumPy Array and Apply astype Method
train_class = np.array(train_class)
test_class = np.array(test_class)

# One-Hot Encoding
y_train_class = to_categorical(train_class.astype('float32'))
y_test_class = to_categorical(test_class.astype('float32'))

def accuracy_func(features, train_EMR, test_EMR, train_X, test_X, y_train_class, y_test_class):
    X_train_EMR = tf.convert_to_tensor(train_EMR[:, features])
    X_test_EMR = tf.convert_to_tensor(test_EMR[:, features])
    
    clf = attention_model(len(X_train_EMR[0]))
    
    Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    clf.fit([train_X, X_train_EMR], y_train_class, validation_split=0.2, epochs=50, batch_size=20, verbose=1, callbacks=[Early_stop])
    
    pred = clf.predict([test_X, X_test_EMR])
    pred_arg = np.argmax(pred, axis=1)
    
    y_test_class_arg = np.argmax(y_test_class, axis=1)
    
    f1 = f1_score(y_test_class_arg, pred_arg, average='macro')
    return f1



# Q-Learning and Model Training
Q_values = np.array([[-1, -1]] * 10)

all_rewards, updated_Q_values = q_learning(
    num_episodes=50, 
    num_agents=10, 
    Q_values=Q_values, 
    accuracy_func=accuracy_func, 
    train_EMR=train_EMR, 
    test_EMR=test_EMR, 
    train_X=train_X, 
    test_X=test_X, 
    y_train_class=y_train_class, 
    y_test_class=y_test_class
)
