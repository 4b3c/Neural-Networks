import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1.0, 2.0, 3.0, 2.5],
	 [2.0, 5.0, -1.0, 2.0],
	 [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		
class Activation_RelU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

Layer1 = Layer_Dense(2, 5)
Activation1 = Activation_RelU()

Layer1.forward(X)
Activation1.forward(Layer1.output)
print(Activation1.output)
