# Andrew Ng machine learning homework in Python
# Homework 1
#                                                                 written by Soyoung, Yang
#                                                                       2019.10.13  v1.0.0
# Requirements: re, numpy, matplotlib.
# Needs 'ex1data1.txt' file
#
# Used numpy's array as a vector/matrix representation.
# Not used any Pytorch or Tensorflow things.


import numpy as np
import re
import matplotlib.pyplot as plt


# 1

def warmUpExercise(a):
    # return identity matrix, size of A
    i = np.identity(a)
    return i

# A = warmUpExercise(5)
# print(A)



# 2

with open('data/ex1data1.txt',"r") as f:
    data1_x = []
    data1_y = []
    lines = f.readlines()
    for line in lines:
        x, y = line.split(',')
        y = re.sub(r'\n', '', y)
        data1_x.append(np.float(x))
        data1_y.append(np.float(y))

    m = len(data1_y) # length of data

# Change list datatype To numpy array.
data1_x = np.asarray(data1_x)
data1_y = np.asarray(data1_y)


## 2.1. Plotting data

def plot_data(x, y, p):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y, p)
    ax.set_xlabel('Profit in $10,000s')
    ax.set_ylabel('Population of City in 10,000s')
    plt.show()
    # plt.pause(5)
    # plt.close()

# plot_data(data1_x, data1_y, 'rx')


## 2.2 Grdient Descent, ComputeCost

def computeCost(X, y, theta):
    m = len(y)
    J = 0
    j = np.dot(X, theta) - y
    jj = np.multiply(j, j)
    J = (1 / (2*m)) * np.sum(jj)

    return J

def gradientDescent(X, y, theta, alpha, num_iters):

    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters-1): # take -1 because range starts from 0

        error = np.dot(X, theta) - y
        x1 = np.multiply(error, X[0,:])
        x2 = np.multiply(error, X[1,:])
        temp0 = theta[0] - ( alpha /m ) * np.sum(x1)    # array value, like '[value]'
        temp0 = np.float(temp0)                         # so get out of the real value, like 'value'
        temp1 = theta[1] - ( alpha /m ) * np.sum(x2)
        temp1 = np.float(temp1)
        theta = np.array([[temp0], [temp1]])            # reshape array to 2 x 1

        # X = 97 x 2 , h-y = 97 x 1 => 2 x 1
        XX = X.transpose()
        h = np.dot(X, theta) - y
        new_theta = theta - alpha * (1/m) * np.dot(XX, h)
        theta = np.array(new_theta)

        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history

X = np.array([np.zeros(m), data1_x[:]])     # 2 x 97
X = X.transpose()                           # 97 x 2
y = np.array([data1_y])                     # 1 x 97
y = y.transpose()                           # 97 x 1
theta = np.zeros([2, 1])                    # 2 x 1

iterations = 1500
alpha = 0.01

cost = computeCost(X, y, theta)
print('Initial cost is ... ', cost)

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ', theta[0], theta[1])


## 2.3. Debugging

def plot_regression(X, y, theta):
    fig = plt.figure()
    ax = fig.add_subplot()
    x = X[:,1]
    yhat = np.dot(X,theta)
    ax.plot(x, yhat, 'b-')
    ax.plot(x, y, 'rx')
    plt.show()
    # plt.pause(5)
    # plt.close()

plot_regression(X, y, theta)







