import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
class Loss_CC(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)
        print("cliped")
        print(y_pred_clip)
        if len(np.array(y_true).shape) == 1:
            correct_confidenc = y_pred_clip[range(samples), y_true]
        elif len(np.array(y_true).shape) == 2:
            correct_confidenc = np.sum(y_pred_clip*y_true, axis=1)
            print("confidences")
            print(y_pred_clip*y_true)
        print(correct_confidenc)
        negative_likelihood = -np.log(correct_confidenc)
        return negative_likelihood

X, y = spiral_data(samples=32, classes=1)

X = [[-5, -3, -4, -10, -12, -4, -3, -5,
     -1, -1, -1, -1, -1, -1, -1, -1,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1,
     5, 3, 4, 10, 12, 4, 3, 5],
     [-5, -3, -4, -10, -12, -4, -3, -5,
     -1, -1, -1, -1, -1, -1, -1, -1,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 1, 1, 1,
     5, 3, 4, 10, 12, 4, 3, 5]]

y = [[-5, -3, -4, -10, -12, -4, -3, -5,
     -1, -1, -1, -1, -1, -1, -1, -1,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 1, 1, 1,
     5, 3, 4, 10, 12, 4, 3, 5],
     [-5, -3, -4, -10, -12, -4, -3, -5,
     -1, -1, -1, 0, -1, -1, -1, -1,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, -1, 0, 0, 0,
     0, 0, 0, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 1, 1, 1,
     5, 3, 4, 10, 12, 4, 3, 5]]

dense1 = Layer_Dense(64, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)
activation2 = Activation_Softmax()
dense3 = Layer_Dense(64, 64)
activation3 = Activation_Softmax()
loss_func = Loss_CC()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

print(activation3.output)

loss = loss_func.calculate(activation3.output, y)

print(loss)