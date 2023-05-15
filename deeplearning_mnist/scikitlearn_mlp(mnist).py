# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:15:30 2023

@author: 82107
"""

from sklearn.datasets import load_mnist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# iris 데이터 로드
mnist = load_mnist()

# feature와 label 분리
X = mnist.data
y = mnist.target

# train/test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP 모델 정의
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=10000, random_state=42)

# 모델 학습
mlp.fit(X_train, y_train)

# 예측 결과 출력
y_pred = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))