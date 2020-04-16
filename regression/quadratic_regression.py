# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-04-17 00:51:13
# @Last modified by:   Duc Khai Tong
# @Last Modified time: 2020-04-17 00:55:36

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('data_square.csv').values
# print(data)
N = data.shape[0]
x = data[:, 0].reshape(-1, 1) # -1 This parameter means that its dimension is temporarily unindentified
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('area')
plt.ylabel('price')

tmp = np.zeros((N, 1))
for i in range(0, N):
	tmp[i] = x[i] * x[i]

x = np.hstack((np.ones((N, 1)), x)) # Stack arrays in sequence horizontally
x = np.hstack((x, tmp))

w = np.array([1977., -97., 0.97]).reshape(-1, 1) # randomly generated ?

numOfIteration = 10000
cost = np.zeros((numOfIteration, 1)) # loss function
learning_rate = 0.0000000001

for i in range(1, numOfIteration):
	r = np.dot(x, w) - y
	cost[i] = 0.5 * np.sum(r * r)
	w[0] -= learning_rate * np.sum(r)
	w[1] -= learning_rate * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
	w[2] -= learning_rate * np.sum(np.multiply(r, x[:, 2].reshape(-1, 1)))
	print(cost[i])

# result = np.array([2000., -100., 1]).reshape(-1, 1)
predict = np.dot(x, w)


plt.scatter(x[:, 1], predict)
plt.show()