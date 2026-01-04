import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Reading the data and storing it in a variable after dropping the empty parts
data = pd.read_csv("Salary_Data.csv")
data = data.dropna(subset=['Years of Experience', 'Salary'])

#Assigning the data to the training and target variables
X = data['Years of Experience'].values
y = data['Salary'].values
#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
#Converting them into Pytorch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

#Using standard scaling on the data
def Scaling(X_train, y_train):
    X_train_Scaled = (X_train-X_train.mean())/X_train.std()
    y_train_Scaled = (y_train-y_train.mean())/y_train.std()
    return X_train_Scaled, y_train_Scaled
X_train_Scaled, y_train_Scaled = Scaling(X_train, y_train)

#Defining gradeint descent using Pytorch's autograd feature and finding the scaled values of the requred parameters
def gradient_descent(X_train_Scaled, y_train_Scaled):

    theta0_scaled = torch.zeros(1, requires_grad=True, dtype=X_train_Scaled.dtype)
    theta1_scaled = torch.zeros(1, requires_grad=True, dtype=X_train_Scaled.dtype)

    for epoch in range(5000):
        y_pred = theta0_scaled + theta1_scaled * X_train_Scaled
        loss = (((y_pred - y_train_Scaled)**2)/2).mean()
        loss.backward()
        with torch.no_grad():
            theta0_scaled.data -= 0.01*theta0_scaled.grad
            theta1_scaled.data -= 0.01*theta1_scaled.grad
        theta1_scaled.grad.zero_()
        theta0_scaled.grad.zero_()
        if epoch%50 == 0:
            print(f"Epoch = {epoch}\nMSE = {loss}")
    return theta0_scaled.item(), theta1_scaled.item()
theta0_scaled, theta1_scaled = gradient_descent(X_train_Scaled, y_train_Scaled)

#Descaling the paramameters (keep in mind that pytorch std and mean returns tesnors unlike numpy arrays that return the values for 1D arrays)
def descaling(theta0_scaled, theta1_scaled):
    theta1_original = theta1_scaled * y_train.std().item() / X_train.std().item()
    theta0_original = theta0_scaled * y_train.std().item() + y_train.mean().item() - theta1_scaled * y_train.std().item() * X_train.mean().item() / X_train.std().item()
    return theta0_original, theta1_original
theta0, theta1 = descaling(theta0_scaled, theta1_scaled)

    
plt.scatter(X_train, y_train, alpha=0.2)
x = torch.linspace(1,30, steps=5)
def y_pred(x):
    y = theta0 + theta1*x
    return y
print(f"Salary = {theta0} + {theta1} x Years of Experience")
plt.plot(x,y_pred(x), color='r', linewidth = 2)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

