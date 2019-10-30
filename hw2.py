
# Andrew Ng machine learning homework in Python
# Homework 2
#                                                                 written by Soyoung, Yang

import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import torch.optim as optim

# Load data
with open('data/ex2data1.txt',"r") as f:
    data1_x1 = []
    data1_x2 = []
    data1_y = []
    lines = f.readlines()
    for line in lines:
        x1, x2, y = line.split(',')
        y = re.sub(r'\n', '', y)    # delete '/n' in y
        data1_x1.append(np.float(x1))
        data1_x2.append(np.float(x2))
        data1_y.append(np.float(y))

X = np.array([data1_x1, data1_x2])  # 2 x 100
X = X.transpose()                   # 100 x 2, make column space as feature space
y = np.array([data1_y])
y = y.transpose()


# PART 1 : Plotting

# print("Plotting data with + indicating (y = 1) examples and o, indicating (y = 0) examples.\n")

def plotData(X, Y):
    # find each label data's location
    posloc = []
    negloc = []
    i = 0
    for yy in Y:
        if yy == 1:
            posloc.append(i)
        else:
            negloc.append(i)
        i += 1
    # divide pos/neg data X
    posX = []
    negX = []
    for loc in posloc:
        posX.append(X[loc, :])
    for loc in negloc:
        negX.append(X[loc, :])
    posX = np.array(posX)       # 60 x 2
    negX = np.array(negX)       # 40 x 2

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(posX[:,0], posX[:,1], 'k+')
    ax.plot(negX[:,0], negX[:,1], 'ko')
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    plt.show()
    plt.close()
    return posX, negX

posX, negX = plotData(X, y)


# PART 2: Compute Cost and Gradient

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# print('sigmoid(0) = ', sigmoid(np.array([[0,0],[0,1]])))


def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    z = sigmoid(np.dot(X, theta))
    J = (-1/m) * np.sum( np.multiply(y, np.log(z)) + np.multiply((1-y), np.log(1-z)))
    temp = sigmoid(np.dot(X,theta))
    error = temp - y
    grad = (1/m) * (np.dot(X.transpose(), error))
    return J, grad

m, n = X.shape
X = np.array([np.ones(m), data1_x1, data1_x2])
X = X.transpose()

initial_theta = np.zeros([n+1, 1])
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): ', cost)
print('Gradient at initial theta (zeros): \n', grad)


# PART 3 : Optimize theta
def compute_cost(theta, X, y):  # computes cost given predicted and actual values
    m,n = X.shape  # number of training examples
    theta = theta.reshape([n,1])
    z = sigmoid(np.dot(X,theta))
    z = z.reshape([m,1])
    J = (-1 / m) * np.sum(np.multiply(y, np.log(z)) + np.multiply((1 - y), np.log(1 - z)))
    return J

def compute_grad(theta, X, y):
    # print theta.shape
    m, n = X.shape
    grad = theta.reshape([n,1])
    temp = sigmoid(np.dot(X, theta))
    temp = temp.reshape([m,1])
    error = temp - y
    error = error.reshape([m,1])
    for i in range(n-1):
        grad[i] = (1/m) * (np.dot(X[:,i], error))
    return grad.flatten()

result = opt.fmin_bfgs(f=compute_cost, x0=initial_theta, args=(X, y),fprime = compute_grad)
print('optimized theta is: ', result)

opt_theta = result

def plotDecisionBoundary(theta, X, posX, negX):
    # plot_data
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(posX[:, 0], posX[:, 1], 'k+', label='Admitted')
    ax.plot(negX[:, 0], negX[:, 1], 'ko', label='Not admitted')
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')

    plot_x = [min(X[:,1])-2, max(X[:,1]+2)]
    plot_y = (-1) * (1/theta[2]) * (np.multiply(theta[1], plot_x) + theta[0])
    ax.plot(plot_x, plot_y, label='Decision Boundary')
    ax.legend(loc='upper left')
    plt.show()

plotDecisionBoundary(opt_theta, X, posX, negX)


# Part 4

prob = sigmoid(np.dot([1, 45, 85], opt_theta))
print('\nFor a student with scores 45 and 85, we predict an admission: ', prob)


def predict(theta, X):
    m, _ = X.shape
    p = np.zeros([m, 1])
    p = np.round(sigmoid(np.dot(X, theta)))
    return p

p = predict(opt_theta, X)
print('Train Accuract: {}\n'.format(np.mean(p==y)*100))

# Train Accuract: 52.0
# fmin_bfgs function doesn't find accurate optima.


