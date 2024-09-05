import random
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

def get_reward(features, accuracy_func, train_EMR, test_EMR, train_X, test_X, y_train_class, y_test_class):
    if not features:
        return 0
    acc = accuracy_func(features, train_EMR, test_EMR, train_X, test_X, y_train_class, y_test_class) * 100
    tot_f = len(features)
    R = acc
    if tot_f > K_len:
        R = acc * K_len / tot_f
    return R


def q_learning(num_episodes, num_agents, Q_values, accuracy_func, train_EMR, test_EMR, train_X, test_X, y_train_class, y_test_class):
    epsilon = 0.4
    alpha = 0.2
    epsilon_decay_rate = 0.995
    alpha_decay_rate = 0.995
    all_rewards = []
    reward_store = {i: 0 for i in range(num_agents)}
    reward_store_array = []
    agent_reward_store_array = []

    with tf.device('/CPU:0'):
        for episode in range(num_episodes):
            print("num_episodes:", episode)
            
            # 각 에이전트의 행동 선택
            actions = [
                np.argmax(Q_values[agent]) if random.uniform(0,1) > epsilon else random.choice([0,1]) 
                for agent in range(num_agents)
            ]
    
            total_features = [i for i, act in enumerate(actions) if act == 1]
            R = get_reward(total_features, accuracy_func, train_EMR, test_EMR, train_X, test_X, y_train_class, y_test_class)
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
    return all_rewards, Q_values
