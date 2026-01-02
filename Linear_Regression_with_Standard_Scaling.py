import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
data = data.dropna(subset= ["Years of Experience","Salary"])
x_train = data['Years of Experience'].values
y_train = data['Salary'].values

#Mean Squared Error function
def mse(x, y, theta1, theta0):
    return (np.mean((y - theta1*x - theta0)**2))/2
#Gradeint function wrt theta1
def grad_theta1(x, y, theta1, theta0):
    return -np.mean(x*(y - theta1*x - theta0))
#Gradient function wrt theta0
def grad_theta0(x, y, theta1, theta0):
    return -np.mean(y - theta1*x - theta0)

def gradient_descent(x, y, L, epochs):
    theta0 = 0
    theta1 = 0
    for i in range(epochs):
        grad1 = grad_theta1(x, y, theta1, theta0)
        grad0 = grad_theta0(x, y, theta1, theta0)
        theta0 = theta0 - L*grad0
        theta1 = theta1 - L*grad1
        
        if i%50 == 0:
            print(f'epoch = {i}')
            print(f'mean squared error = {mse(x, y, theta1, theta0)}')
    return theta0, theta1

x_train_mean = x_train.mean()
y_train_mean = y_train.mean()
x_train_std = x_train.std()
y_train_std = y_train.std()

x_train_normalized = (x_train - x_train_mean)/x_train_std
y_train_normalized = (y_train - y_train_mean)/y_train_std

theta0_normalized, theta1_normalized = gradient_descent(x_train_normalized,y_train_normalized, L=0.01, epochs=500)
plt.scatter(x_train_normalized, y_train_normalized, alpha=0.2)
a = np.linspace(-1,4)
b = theta1_normalized*a + theta0_normalized
plt.plot(a,b, linewidth = 2, color = 'r')
plt.show()
theta1_original = theta1_normalized * y_train_std / x_train_std
theta0_original = theta0_normalized * y_train_std + y_train_mean - theta1_normalized * y_train_std * x_train_mean / x_train_std
c = np.linspace(1,30)
d = theta0_original + theta1_original*c
plt.plot(c,d, linewidth = 2, color = 'r' )
plt.scatter(x_train, y_train, alpha=0.2)
plt.show()
