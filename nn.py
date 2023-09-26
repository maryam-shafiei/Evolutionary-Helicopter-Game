import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        self.params = {
            'w1': np.random.normal(size=(layer_sizes[1], layer_sizes[0])),  # 10x4
            'w2': np.random.normal(size=(layer_sizes[2], layer_sizes[1])),  # 2x10
            'b1': np.zeros((layer_sizes[1], 1)),
            'b2': np.zeros((layer_sizes[2], 1))
        }
        self.layer_sizes = layer_sizes
    def activation(self, x):
        
        # TODO
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        a = [None] * 3
        z = [None] * 3

        a[0] = x  # 4x1

        # input layer to hidden layer
        z[1] = (self.params['w1'] @ a[0]) + self.params['b1']  # 16x1
        a[1] = self.activation(z[1])

        # hidden layer to output layer
        z[2] = (self.params['w2'] @ a[1]) + self.params['b2']  # 16x1
        a[2] = self.activation(z[2])

        return a[2]
