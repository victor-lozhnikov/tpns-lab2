from layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.normal(0, 2 ** -0.5, (input_size, output_size))

    def forward(self, input_data):
        self.input = input_data

        for i in range(len(input_data[0])):
            if np.isnan(input_data[0, i]):
                input_data[0, i] = 0.0

        # print(input_data.shape, self.weights.shape)
        output = np.matmul(input_data, self.weights)
        # print(output)
        return output

    def backward(self, input_error, learning_rate):
        # print(input_error.shape, self.weights.shape)
        error = np.matmul(input_error, self.weights.T)
        # print(self.input.shape, input_error.shape)
        weights_error = np.matmul(self.input.T, input_error)

        for i in range(len(self.input[0])):
            if np.isnan(self.input[0, i]):
                weights_error[0, i] = 0

        self.weights += learning_rate * weights_error
        return error
