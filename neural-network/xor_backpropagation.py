# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-04-15 16:40:37
# @Last modified by:   Duc Khai Tong
# @Last Modified time: 2020-04-17 00:12:50

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputLayer = np.array([[0,0],[0,1],[1,0],[1,1]])
expectedOutput = np.array([[0],[1],[1],[0]])
plt.scatter([0, 1], [0, 1], c='red', edgecolor='none', s=30)
plt.scatter([0, 1], [1, 0], c='blue', edgecolor='none', s=30)
plt.xlabel('x1')
plt.ylabel('x2')



epochs = 10000
learningRate = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

# Random weights and bias initialization
hiddenWeights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hiddenBias = np.random.uniform(size=(1, hiddenLayerNeurons))
outputWeights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
outputBias = np.random.uniform(size=(1, outputLayerNeurons))

print("Initial hidden weights: ",end='')
print(*hiddenWeights)
print("Initial hidden biases: ",end='')
print(*hiddenBias)
print("Initial output weights: ",end='')
print(*outputWeights)
print("Initial output biases: ",end='')
print(*outputBias)

# Training algorithm
for _ in range(epochs):
	# Feedforward
	hiddenLayerActivation = np.dot(inputLayer, hiddenWeights) + hiddenBias
	hiddenLayerOutput = sigmoid(hiddenLayerActivation)
	outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights) + outputBias
	predictedOutput = sigmoid(outputLayerActivation)

	# Backpropagation vs Logistic loss function
	loss = -np.sum(np.multiply(expectedOutput, np.log(predictedOutput)) + np.multiply(1 - expectedOutput, np.log(1 - predictedOutput)))
	print(loss)
	

	# # Updating weights and biases
	# outputWeights -= hiddenLayerOutput.T.dot(predictedOutput - expectedOutput) * learningRate  
	# outputBias -= np.sum(predictedOutput - expectedOutput, axis=0, keepdims=True) * learningRate
	# hiddenWeights -= inputLayer.T.dot(((predictedOutput - expectedOutput).dot(outputWeights.T)) * sigmoid_derivative(hiddenLayerOutput)) * learningRate
	# hiddenBias += np.sum( (((predictedOutput - expectedOutput).dot(outputWeights.T)) * hiddenLayerOutput), axis=0, keepdims=True ) * learningRate


	# Backpropagation vs MSE loss function
	error = expectedOutput - predictedOutput
	derivative_predictedOutput = error * sigmoid_derivative(predictedOutput)
	error_hiddenLayer = derivative_predictedOutput.dot(outputWeights.T)
	derivative_hiddenLayer = error_hiddenLayer * sigmoid_derivative(hiddenLayerOutput)

	# Updating weights and biases
	outputWeights += hiddenLayerOutput.T.dot(derivative_predictedOutput) * learningRate
	outputBias += np.sum(derivative_predictedOutput, axis=0, keepdims=True) * learningRate
	hiddenWeights += inputLayer.T.dot(derivative_hiddenLayer) * learningRate		
	hiddenBias += np.sum(derivative_hiddenLayer, axis=0, keepdims=True) * learningRate

print("Final hidden weights: ",end='')
print(*hiddenWeights)
print("Final hidden bias: ",end='')
print(*hiddenBias)
print("Final output weights: ",end='')
print(*outputWeights)
print("Final output bias: ",end='')
print(*outputBias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predictedOutput)

plt.show()


