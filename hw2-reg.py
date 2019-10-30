# Andrew Ng machine learning homework in Python
# Homework 2 - reg
#                                                                 written by Soyoung, Yang

import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

with open('data/ex2data2.txt','r') as f:
    data1_x1 = []
    data1_x2 = []
    data1_y = []
    lines = f.readlines()
    for line in lines:
        x1, x2, y = line.split(',')
        y = re.sub(r'\n', '', y)  # delete '/n' in y
        data1_x1.append(np.float(x1))
        data1_x2.append(np.float(x2))
        data1_y.append(np.float(y))

X = np.array([data1_x1, data1_x2])  # 2 x 100
X = X.transpose()  # 100 x 2, make column space as feature space
y = np.array([data1_y])
y = y.transpose()

class plot():
    def __init__(self, X, Y):

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
        self.posX = np.array(posX)       # 60 x 2
        self.negX = np.array(negX)       # 40 x 2

    def plot_data(self):
        # plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.posX[:,0], self.posX[:,1], 'k+', label='y=1')
        ax.plot(self.negX[:,0], self.negX[:,1], 'yo', label='y=0')
        ax.set_xlabel('Microchip Test 1')
        ax.set_ylabel('Microchip Test 2')
        # plt.show()

    def plot_DB(self, theta):
        # plot decision boundary
        theta = theta.reshape([theta.size, 1])
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((np.size(u), np.size(v)))
        for i in np.arange(np.size(u)):
            for j in np.arange(np.size(v)):
                z[i, j] = np.sum(mapFeature(u[i], u[j]) * theta)

        z = z.transpose()
        plt.contour(u, v, z)
        plt.show()
        plt.close()

p = plot(X, y)
p.plot_data()

# PART 1

def mapFeature(X1, X2):
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i+1):
            Xij = X1**(i-j)
            Xj = X2**j
            out.append(np.multiply(Xij, Xj))

    out = np.matrix(out)
    return out

X = mapFeature(X[:,0], X[:,1])

n, m = X.shape
initial_theta = np.zeros([n, 1])

lambdaa = 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunctionReg(theta, X, y, lam):
    n, m = X.shape
    theta = theta.reshape([n, 1])

    J = 0
    grad = np.zeros(theta.shape)

    tempTheta = theta
    tempTheta[0] = 0

    z = sigmoid(X.transpose() * theta)
    z = z.reshape([m, 1])
    J = (-1 / n) * np.sum(np.multiply(y, np.log(z)) + np.multiply((1 - y), np.log(1 - z))) \
        + (lam/(2*n)) * np.sum(tempTheta**2)

    return J

def gradFunctionReg(theta, X, y, lam):
    m, n = X.shape
    theta = theta.reshape([m, 1])
    grad = np.zeros([m])

    # tempTheta = theta
    # tempTheta[0] = 0
    temp = sigmoid(X.transpose() * theta)
    error = temp - y
    # grad = (1/n) * (X * error) + (lam/n) * tempTheta # wrong answer for code
    for i in range(m):
        ex = np.multiply(X[i,:], error)
        if i==0 :   # for bias one
            grad[i] = np.sum(ex) / n
        else:
            grad[i] = (np.sum(ex) / n) + lam/n * theta[i,:]
    return grad.flatten()

J = costFunctionReg(initial_theta, X, y, lambdaa)
grad = gradFunctionReg(initial_theta, X, y, lambdaa)
# print(J, grad)


# PART 2

result = opt.minimize(fun=costFunctionReg, jac=gradFunctionReg, x0=initial_theta, args=(X, y, lambdaa))
opt_theta = result.x

p.plot_DB(opt_theta)
# suck

def predict(theta, X):
    m, _ = X.shape
    p = np.zeros([m, 1])
    p = np.round(sigmoid(np.dot(X.transpose(), theta)))
    return p

p = predict(opt_theta, X)
print('Train Accuract: {}\n'.format(np.mean(p==y)*100))

# Train Accuract: 50.574547543809246
# suck scipy
