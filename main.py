import csv
from random import shuffle
import numpy as np

from network import NeuralNetwork
from fully_connected_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from losses import mse
from activations import sigmoid, derivative_sigmoid


with open('ready_data.csv') as file:
    reader = csv.reader(file, delimiter=';')
    data = list(reader)[1:]

shuffle(data)

data_size = len(data)
train_size = int(0.7 * data_size)
test_size = data_size - train_size

x_train = []
y_train = []
x_test = []
y_test = []

for j in range(len(data[0])):
    min_i = 1e9
    max_i = -1e9
    for i in range(len(data)):
        if data[i][j] == '':
            data[i][j] = np.nan
            continue
        data[i][j] = float(data[i][j])
        min_i = min(min_i, data[i][j])
        max_i = max(max_i, data[i][j])
    for i in range(len(data)):
        if data[i][j] != np.nan:
            data[i][j] = (data[i][j] - min_i) / (max_i - min_i) - 0.5

for i in range(len(data)):
    if i < train_size:
        x_train.append(np.asmatrix(data[i][:-2]))
        y_train.append(np.asmatrix(data[i][-2:]))
    else:
        x_test.append(np.asmatrix(data[i][:-2]))
        y_test.append(np.asmatrix(data[i][-2:]))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = NeuralNetwork()
model.set_loss(mse)
model.add(FullyConnectedLayer(15, 60))
model.add(ActivationLayer(sigmoid, derivative_sigmoid))
model.add(FullyConnectedLayer(60, 47))
model.add(ActivationLayer(sigmoid, derivative_sigmoid))
model.add(FullyConnectedLayer(47, 2))

model.fit(x_train, y_train, x_test, y_test, 200, 0.01)
