from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
import numpy as np


class Dense_Layer:
  def __init__(self, n_inputs, n_neurons):
    self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
    self.biases = np.zeros((1, n_neurons)) * 0.01
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

class Actication_function:
    def relu(self, inputs):
        self.output = np.maximum(0, inputs)

    def sigmoid(self, inputs):
        self.output = 1 / ( 1 + np.exp(-inputs) )

    def softmax(self, inputs):
        exp =  np.exp(inputs)
        prop = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = prop

X, y = spiral_data(samples = 5, classes = 3)
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

# Test #
# dense_layer1 = Dense_Layer(2, 3)
# dense_layer1.forward(X)
# print(dense_layer1.output)
########

d1 = Dense_Layer(2,3)
d2 = Dense_Layer(3,3)
d3 = Dense_Layer(3,2)
af = Actication_function()
d1.forward(X)
af_relu = af.relu(d1.output)
print(af.output)
af_sigmoid = af.sigmoid(d1.output)
print(af.output)
af_softmax = af.softmax(d1.output)
print(af.output)
