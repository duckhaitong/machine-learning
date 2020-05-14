# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-04-17 01:36:35
# @Last modified by:   Duc Khai Tong
# @Last Modified time: 2020-05-14 12:57:55

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)


class NeuralNetwork:
	def __init__(self, layers, alpha): # layers: neurons per each layer
		# model layers
		self.layers = layers

		# learning rate
		self.alpha = alpha

		# parameters: weight, bias
		self.W = []
		self.B = []

		for i in range(0, len(layers) - 1):
			self.W.append(np.random.randn(layers[i], layers[i+1])) # random values in a given shape
			self.B.append(np.zeros((layers[i+1], 1)))

	def __repr__(self):
		return "Neural Network [{}]".format("-".join(str(l) for l in self.layers))


	def partial_fit(self, X, y):
		# activation
		A = [X]


		# feedforward
		output = A[0]
		for i in range(0, len(self.layers) - 1):
			output = sigmoid(output.dot(self.W[i]) + (self.B[i].T))
			A.append(output)

		# backpropagation
		# y = y.reshape(-1, 1)
		dA = [-(y/A[-1] - (1 - y)/(1 - A[-1]))]
		dW = []
		dB = []
		for i in reversed(range(0, len(self.layers) - 1)):
			dw = A[i].T.dot(dA[-1] * sigmoid_derivative(A[i+1]))
			db = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1, 1) 
			da = dA[-1] * sigmoid_derivative(A[i+1]).dot(self.W[i].T)

			dW.append(dw)
			dB.append(db)
			dA.append(da)
		
		# reverse
		dW = dW[::-1]
		dB = dB[::-1]

		# gradient descent
		for i in range(0, len(self.layers) - 1):
			self.W[i] -= self.alpha * dW[i]
			self.B[i] -= self.alpha * dB[i]


	# predict
	def predict(self, X):
		for i in range(0, len(self.layers) - 1):
			X = sigmoid(X.dot(self.W[i]) + self.B[i].T)
		return X

	# calculating the loss function
	def calculate_loss(self, X, y):
		y_predict = self.predict(X)
		return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))

	def fit(self, X, y, epochs, verbose):
		for epoch in range(0, epochs):
			self.partial_fit(X, y)
			if epoch % verbose == 0:
				loss = self.calculate_loss(X, y)
				print("Epoch {}, loss {}".format(epoch, loss))


if __name__ == "__main__":
	# read dataset
	data = pd.read_csv('xor.csv').values
	N, d = data.shape
	x = data[:, 0:d-1].reshape(-1, d-1)
	y = data[:, 2].reshape(-1, 1)

	p = NeuralNetwork([x.shape[1], 2, 1], 0.1)
	p.fit(x, y, 10000, 100)
	print(p)
	print(p.predict(x))
	