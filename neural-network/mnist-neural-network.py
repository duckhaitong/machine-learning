# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-04-30 03:04:29
# @Last modified by:   Duc Khai Tong
# @Last Modified time: 2020-05-03 15:47:13

import numpy as np
import matplotlib.pyplot as plt
from activations import sigmoid
from activations import sigmoid_derivative
from keras.datasets import mnist


def load_data()

class NeuralNetwork:
	def __init__(self, layers, alpha):
		# Initialize a Neural Network model
		# layers : list, optional
		# 	A list of intergers specifying number of neurons in each layer
		# alpha : float, optional
		# 	Learning rate for gradient descent optimization
		self.layers = layers
		self.alpha = alpha
		self.W = []
		self.B = []

		for i in range(0, len(layers) - 1):
			self.W.append(np.random.randn(layers[i], layers[i+1]))
			self.B.append(np.zeros((layers[i+1], 1)))

	def __repr__(self):
		return "Neural Network [{}]".format("-".join(str(l) for l in self.layers))

	def partial_fit(self, x, y):
		# activation
		A = [x] 

		# feedforward
		out = A[-1]
		for i in range(0, len(self.layers) - 1):
			out = sigmoid(out.dot(self.W[i]) + (self.B[i].T))
			A.append(out)

		# backpropagation
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
	# Setup train and test splits
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
	print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28 pixels
	print(y_test)
	# validation set
	x_val, y_val = x_train[50000:60000,:], y_train[50000:60000]
	# train set
	x_train, y_train = x_train[:50000,:], y_train[:50000]
	print(x_train.shape)
	# print(x_train)
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	# print(x_train)

	p = NeuralNetwork([2, 2, 1], 0.1)
	print(p)
