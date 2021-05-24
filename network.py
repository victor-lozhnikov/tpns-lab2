import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def predict(self, input_data):
        result = []
        for sample in input_data:
            output = sample
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        train_error = []
        test_error = []
        for epoch in range(epochs):
            for i in range(len(x_train)):
                # print(x_train[i])
                # print(y_train[i])
                forward_output = x_train[i]

                for layer in self.layers:
                    forward_output = layer.forward(forward_output)

                # print(forward_output)

                backward_error = np.array([0.0, 0.0])

                if not np.isnan(y_train[i][0][0]):
                    backward_error[0] = y_train[i, 0, 0] - forward_output[0, 0]
                if not np.isnan(y_train[i][0][1]):
                    backward_error[1] = y_train[i, 0, 1] - forward_output[0, 1]

                backward_error = np.asmatrix(backward_error.T)

                for layer in reversed(self.layers):
                    backward_error = layer.backward(backward_error, learning_rate)

            cur_train_error = 0.0
            y_pred = self.predict(x_train)
            for i in range(len(y_train)):
                cur_train_error += self.loss(y_train[i, 0], y_pred[i])
            cur_train_error /= len(x_train)
            train_error.append(cur_train_error)

            cur_test_error = 0.0
            y_pred = self.predict(x_test)
            for i in range(len(y_test)):
                cur_test_error += self.loss(y_test[i, 0], y_pred[i])
            cur_test_error /= len(x_test)
            test_error.append(cur_test_error)

            print('epoch ' + str(epoch) + ': train error = ' + str(cur_train_error) + ', test error = ' + str(cur_test_error))

        x_axis = np.arange(0, epochs)
        plt.plot(x_axis, train_error)
        plt.plot(x_axis, test_error)
        plt.legend(["train", "test"])
        plt.show()
