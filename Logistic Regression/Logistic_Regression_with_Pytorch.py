import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Social_Network_Ads.csv")
data = data.dropna(subset=['EstimatedSalary', 'Purchased', 'Age'])
X = torch.FloatTensor(data[['Age', 'EstimatedSalary']].values)
y = torch.FloatTensor(data['Purchased'].values)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

def min_max_scaling(X_train, X_test):
    Xmin = X_train.min(dim=0, keepdim=True).values
    Xmax = X_train.max(dim=0, keepdim=True).values
    X_train_scaled = (X_train - Xmin)/(Xmax - Xmin)
    X_test_scaled = (X_test - Xmin)/(Xmax - Xmin)
    return X_train_scaled, X_test_scaled

def standard_scaling(X_train, X_test):
    Xmean = X_train.mean(dim=0, keepdim=True)
    Xstd = X_train.std(dim=0, keepdim=True)
    X_train_scaled = (X_train - Xmean)/(Xstd)
    X_test_scaled = (X_test - Xmean)/(Xstd)
    return X_train_scaled, X_test_scaled

X_train_standardscaled, X_test_standardscaled = standard_scaling(X_train, X_test)
X_train_minmaxscaled, X_test_minmaxscaled = min_max_scaling(X_train, X_test)
def gradient_descent(X,y):
    losses = []
    theta0 = torch.zeros(1, requires_grad=True)
    theta1 = torch.zeros(2, requires_grad=True)
    for epoch in range(5001):
        z =  X@theta1 + theta0
        h = 1 / (1 + torch.exp(-z)) 
        cost = -(y * torch.log(h) + (1 - y) * torch.log(1 - h)).mean()
        cost.backward()
        with torch.inference_mode():
            theta0.data -= 0.01*theta0.grad
            theta1.data -= 0.01*theta1.grad
        theta1.grad.zero_()
        theta0.grad.zero_()
        losses.append(cost.item())
        if epoch%500 == 0:
            print(f"Cost Function = {cost}\n epoch = {epoch}")
    return theta0.item(), theta1, losses


theta0_stand, theta1_stand, losses_stan = gradient_descent(X_train_standardscaled, y_train)
theta0_minmax, theta1_minmax, losses_minmax = gradient_descent(X_train_minmaxscaled, y_train)

def predict(X, theta0, theta1):
    z = X@theta1 + theta0
    h = 1 / (1 + torch.exp(-z))
    return (h >= 0.5).float()

y_test_pred_stand = predict(X_test_standardscaled, theta0_stand, theta1_stand)
y_test_pred_minmax = predict(X_test_minmaxscaled, theta0_minmax, theta1_minmax)
accuracy_stan = (y_test_pred_stand == y_test).float().mean()*100
accuracy_minmax = (y_test_pred_minmax == y_test).float().mean()*100
print(f"Accuracy with standard scaling = {accuracy_stan}%\nAccuracy with min max scaling = {accuracy_minmax}%")
print(f"Accuracy difference = {accuracy_stan}% - {accuracy_minmax}% = {accuracy_stan - accuracy_minmax}%")

plt.figure(figsize=(9, 6))
plt.plot(losses_stan, label='StandardScaler', linewidth=2, color='r')
plt.plot(losses_minmax, label='MinMaxScaler', linewidth=2, color='b')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.gca().locator_params(nbins=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

