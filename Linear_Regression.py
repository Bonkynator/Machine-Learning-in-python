import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_example.csv')
def cost(x_train, y_train, theta0, theta1):
    m = len(x_train)
    cost = 0
    for _ in range(m):
       current = (y_train[_] - x_train[_]*theta1 - theta0)
       cost = current+cost
    total_cost = cost/(2*m)
    return total_cost

def cost_grad0(x_train, y_train, theta0, theta1):
    m = len(x_train)
    cost_grad0 = 0
    for i in range(m):
        current = -(y_train[i]-x_train[i]*theta1 - theta0)
        cost_grad0 = cost_grad0 + current
    total_grad = cost_grad0/m
    return total_grad

def cost_grad1(x_train, y_train, theta0, theta1):
    m = len(x_train)
    cost_grad1 = 0
    for i in range(m):
        current = -x_train[i]*(y_train[i]-x_train[i]*theta1 - theta0)
        cost_grad1 = cost_grad1 + current
    total_grad = cost_grad1/m
    return total_grad

def gradeint_descent(L, epochs, x_train, y_train):
    theta0 = 0
    theta1 = 0
    for i in range(epochs):
        theta0g = cost_grad0(x_train, y_train, theta0, theta1)
        theta1g = cost_grad1(x_train, y_train, theta0, theta1)
        theta1 = theta1 - L*theta1g
        theta0 = theta0 - L*theta0g
    return theta0, theta1
x_train=data['x'].values
y_train=data['y'].values

L = 0.001
epochs = 1000
print(gradeint_descent(L, epochs, x_train, y_train))
values = (gradeint_descent(L, epochs, x_train, y_train))
m = values[1]
b = values[0]
x = np.linspace(0, 10)
y = m*x + b
def func(x):
    return m*x +b
plt.plot(x,y, linewidth=5)
plt.scatter(data['x'], data['y'])
plt.show()
print(func(11))