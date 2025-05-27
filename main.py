import numpy as np

inputs = [1, 2, 3, 4]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 8)
layer2 = Layer_Dense(8, 2)

layer1.forward(inputs)
layer2.forward(layer1.output)

print(layer1.output)
print(layer2.output)