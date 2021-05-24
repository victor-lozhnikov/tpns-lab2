from layer import Layer
import numpy as np


class ActivationLayer(Layer):
    def __init__(self, activation, derivative_activation):
        super().__init__()
        self.activation = np.vectorize(activation)
        self.derivative_activation = np.vectorize(derivative_activation)

    def forward(self, input_data):
        self.input = input_data
        output = self.activation(self.input)
        return output

    def backward(self, output_error, learning_rate):
        return np.multiply(np.asmatrix(self.derivative_activation(self.input)), output_error)
