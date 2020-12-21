import sys
import numpy as np
from numpy import genfromtxt
import argparse

parser = argparse.ArgumentParser(description='Neural Network')
parser.add_argument('--data', type=str, help='.csv file containing data')
parser.add_argument('--eta', type=float, help='Learning rate')
parser.add_argument('--iterations', type=int, help='Number of epochs')
args = parser.parse_args()

filename = args.data
lr = args.eta
n = args.iterations

df = genfromtxt(filename, delimiter=',')
X = df[:, 0:2]
Y = df[:, -1]

list_w_bias_h1 = [0.2]
list_w_a_h1 = [-0.3]
list_w_b_h1 = [0.4]
list_w_bias_h2 = [-0.5]
list_w_a_h2 = [-0.1]
list_w_b_h2 = [-0.4]
list_w_bias_h3 = [0.3]
list_w_a_h3 = [0.2]
list_w_b_h3 = [0.1]
list_w_bias_o = [-0.1]
list_w_h1_o = [0.1]
list_w_h2_o = [0.3]
list_w_h3_o = [-0.4]

eta = 0.2
num_iterations = 2
a = []
b = []
t = []
a_twice = [float('-0')]
b_twice = [float('-0')]
t_twice = [float('-0')]
h1 = [float('-0')]
h2 = [float('-0')]
h3 = [float('-0')]
list_delta_h1 = [float('-0')]
list_delta_h2 = [float('-0')]
list_delta_h3 = [float('-0')]
list_delta_o = [float('-0')]
o = [float('-0')]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def loss(y_hat, y):
    # L = -(y * np.log10(y_hat) + ((1 - y) * np.log10(1 - y_hat)))# cross entropy loss
    L = ((y_hat - y) ** 2) / 2
    return L


def derivative_loss(y_hat, y):
    # dL =  (y_hat-y)/(y_hat*(1-y_hat)) #cross entropy loss derivative
    dL = (y_hat - y)
    return dL


def train(x, Y, n=2, lr=0.2):
    # initialize weights from the gauss3 solutions


    w_layer1 = np.array([[-0.30000, 0.40000],
                         [-0.10000, -0.40000],
                         [0.20000, 0.10000]])
    b_layer1 = np.array([0.2, -0.5, 0.3])
    w_layer2 = np.array([0.10000, 0.30000, -0.40000])
    b_layer2 = np.array([-0.10000])

    for iterations in range(n):
        for i in range(0, len(x)):
            X = x[i]
            y = Y[i]

            a_twice.append(float(X[0]))
            b_twice.append(float(X[1]))
            t_twice.append(float(y))

            # forward step 1
            z1 = np.dot(w_layer1, X) + b_layer1

            # activation sigmoid
            sigma_1 = sigmoid(z1)

            # forward step 2
            z2 = np.dot(w_layer2, sigma_1) + b_layer2

            # activation sigmoid
            sigma_2 = sigmoid(z2)

            h1.append(sigma_1[0])
            h2.append(sigma_1[1])
            h3.append(sigma_1[2])
            o.append(sigma_2[0])

            Error = loss(sigma_2, y)

            delta_o = sigma_2 * (1 - sigma_2) * (y - sigma_2)
            delta_h1 = sigma_1[0] * (1 - sigma_1[0]) * (w_layer2[0] * delta_o)
            delta_h2 = sigma_1[1] * (1 - sigma_1[1]) * (w_layer2[1] * delta_o)
            delta_h3 = sigma_1[2] * (1 - sigma_1[2]) * (w_layer2[2] * delta_o)
            list_delta_o.append(delta_o[0])
            list_delta_h1.append(delta_h1[0])
            list_delta_h2.append(delta_h2[0])
            list_delta_h3.append(delta_h3[0])

            A = derivative_loss(sigma_2, y) * sigmoid_derivative(sigma_2)

            dL_dw_11 = A * w_layer2[0] * sigmoid_derivative(sigma_1[0]) * X[0]  # dL/dw11
            w_layer1[0][0] = w_layer1[0][0] - lr * dL_dw_11  # parameter update
            list_w_a_h1.append(w_layer1[0][0])

            dL_dw_12 = A * w_layer2[0] * sigmoid_derivative(sigma_1[0]) * X[1]  # dL/dw12
            w_layer1[0][1] = w_layer1[0][1] - lr * dL_dw_12  # parameter update
            list_w_b_h1.append(w_layer1[0][1])

            dL_dw_21 = A * w_layer2[1] * sigmoid_derivative(sigma_1[1]) * X[0]  # dL/dw21
            w_layer1[1][0] = w_layer1[1][0] - lr * dL_dw_21  # parameter update
            list_w_a_h2.append(w_layer1[1][0])

            dL_dw_22 = A * w_layer2[1] * sigmoid_derivative(sigma_1[1]) * X[1]  # dL/dw22
            w_layer1[1][1] = w_layer1[1][1] - lr * dL_dw_22  # parameter update
            list_w_b_h2.append(w_layer1[1][1])

            dL_dw_31 = A * w_layer2[2] * sigmoid_derivative(sigma_1[2]) * X[0]  # dL/dw31
            w_layer1[2][0] = w_layer1[2][0] - lr * dL_dw_31  # parameter update
            list_w_a_h3.append(w_layer1[2][0])

            dL_dw_32 = A * w_layer2[2] * sigmoid_derivative(sigma_1[2]) * X[1]  # dL/dw32
            w_layer1[2][1] = w_layer1[2][1] - lr * dL_dw_32  # parameter update
            list_w_b_h3.append(w_layer1[2][1])

            dL_db1 = A * w_layer2[0] * sigmoid_derivative(sigma_1[0]) * 1  # dL/b1
            b_layer1[0] = b_layer1[0] - lr * dL_db1  # parameter update
            list_w_bias_h1.append(b_layer1[0])

            dL_db2 = A * w_layer2[1] * sigmoid_derivative(sigma_1[1]) * 1  # dL/db2
            b_layer1[1] = b_layer1[1] - lr * dL_db2  # parameter update
            list_w_bias_h2.append(b_layer1[1])

            dL_db3 = A * w_layer2[2] * sigmoid_derivative(sigma_1[2]) * 1  # dL/db3
            b_layer1[2] = b_layer1[2] - lr * dL_db3  # parameter update
            list_w_bias_h3.append(b_layer1[2])

            dL_dw_a = A * sigma_1[0]  # dL/dwa
            w_layer2[0] = w_layer2[0] - lr * dL_dw_a  # parameter update
            list_w_h1_o.append(w_layer2[0])

            dL_dw_b = A * sigma_1[1]  # dL/dwb
            w_layer2[1] = w_layer2[1] - lr * dL_dw_b  # parameter update
            list_w_h2_o.append(w_layer2[1])

            dL_dw_c = A * sigma_1[2]  # dL/dwc
            w_layer2[2] = w_layer2[2] - lr * dL_dw_c  # parameter update
            list_w_h3_o.append(w_layer2[2])

            dL_db4 = A * 1  # dL/db4
            b_layer2[0] = b_layer2[0] - lr * dL_db4  # parameter update
            list_w_bias_o.append(b_layer2[0])


train(X, Y, n, lr)

print(a_twice, b_twice, h1, h2, h3, o, t_twice, list_delta_h1, list_delta_h2,
      list_delta_h3, list_delta_o, list_w_bias_h1, list_w_a_h1, list_w_b_h1, list_w_bias_h2,
      list_w_a_h2, list_w_b_h2, list_w_bias_h3, list_w_a_h3, list_w_b_h3, list_w_bias_o,
      list_w_h1_o, list_w_h2_o, list_w_h3_o)
