# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-04-07 02:44:54
# @Last modified by:   Duc Khai Tong
# @Last Modified time: 2020-04-13 12:31:07

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
# load data
data = pd.read_csv('dataset.csv').values

N, d = data.shape
# print(N, d)
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

x_accept = x[y[:, 0] == 1]
x_refuse = x[y[:, 0] == 0]

plt.scatter(x_accept[:, 0], x_accept[:, 1], c='blue', edgecolors='none', s=30, label='x_accept')
plt.scatter(x_refuse[:, 0], x_refuse[:, 1], c='red', edgecolors='none', s=30, label='x_refuse')
plt.legend(loc=1)
plt.xlabel('salary')
plt.ylabel('experience')

# add "1" column
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0., 1., 1.]).reshape(-1, 1)

numOfIteration = 1000
cost = np.zeros((numOfIteration, 1))
learningRate = 0.01

for i in range(1, numOfIteration):
	y_predict = sigmoid(np.dot(x, w))
	cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
	w = w - learningRate * np.dot(x.T, y_predict - y) # Gradient descent

t = 0.7

# draw model
plt.plot((4, 10), ( -(w[0] + 4 * w[1] + np.log(1/t - 1)) / w[2], -(w[0] + 10 * w[1] + np.log(1/t - 1)) / w[2] ), 'green')

plt.show()