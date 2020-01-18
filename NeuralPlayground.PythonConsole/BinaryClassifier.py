#Inspired by: https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24

import numpy as np
import ActivationFunctions as af


input = np.array([
    [1,0,1], 
    [0,0,1], 
    [0,0,1], 
    [0,0,1],
    [1,0,1],
    [1,0,1],
    [0,0,1]])
output = [1,0,0,0,1,1,0]

class NeuralNetwork:
    
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [.50, .50, .50]

    def feed_forward(self):
        self.hidden = self.apply_sigmoid(self.inputs, self.weights)

    def backpropagation(self):
        #The error is the difference between the correct output and the the predicted one.
        self.error = self.outputs - self.hidden
        #Delta is how much we're going to adjust. Basically, we multiply the error by the derivative of the prediction (i.e., the slope).
        delta = self.error * af.sigmoid_derivative(self.hidden)
        #We adjust the weights by taking the product of delta and the transposed inputs. As of this writting, I'm unsure of the reason why we're taking the T of inputs.
        self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()

    def predict(self, input):
        return self.apply_sigmoid(input, self.weights)

    def apply_sigmoid(self, inputs, weights):
        return af.sigmoid(np.dot(inputs, weights))

NN = NeuralNetwork(input, output)
NN.train()

print("Prediction: ", NN.predict(np.array([[0,0,1]])), " - Correct: 0")

print("Prediction: ", NN.predict(np.array([[1,0,1]])), " - Correct: 1")

print("Prediction: ", NN.predict(np.array([[0,0,0]])), " - Correct: 0")


































