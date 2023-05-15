# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:07:42 2023

@author: 82107
"""

import sys, os
sys.path.append(os.pardir)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from TwoLayerNet import TwoLayerNet
from FiveLayerNet import FiveLayerNet

## MNIST 데이터 읽기
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리
X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255.0
x_train, x_test, t_train, t_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# train/validation 데이터 분리
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2, random_state=42)

#network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = FiveLayerNet(input_size=784, hidden_size1 = 50, hidden_size2= 50, hidden_size3= 50, output_size=10)

iters_num = 10000   #반복횟수
train_size = x_train.shape[0]
batch_size = 100     #배치(묶음)크기
learning_rate = 0.1

train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    #배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    """
    #매개변수 갱신(2층)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    """
    #매개변수 갱신(5층)
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
        network.params[key] -= learning_rate * grad[key]
    
    
    #1epoch 당의 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
     
print("반복 횟수: " + str(iters_num) + ", 묶음의 크기: " + str(batch_size))
print("train_acc: " + str(train_acc) + ", test_acc: " + str(test_acc))