#Valery Lozko
#CPSC 8430 Fall 2024
#HW 1


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-1, 1, 1000).reshape(-1, 1) #defining the tensor for x parameter
y1 = (x**3) - (x**2) #function #1
y2 = torch.cos(x) #function #2

#First Model
class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 15)
        self.fc4 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#Second Model
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 9)
        self.fc6 = nn.Linear(9, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)  # Output layer has no activation (for regression)
        return x

#Third Model
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 29)
        self.fc2 = nn.Linear(29, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#initializing models
Model0_1 = Model0()
Model1_1 = Model1()
Model2_1 = Model2()

Model0_2 = Model0()
Model1_2 = Model1()
Model2_2 = Model2()

#print number of paramaters in the models
print(f'Total number of params in Model0: {sum(p.numel() for p in Model0_1.parameters())}') #526
print(f'Total number of params in Model1: {sum(p.numel() for p in Model1_1.parameters())}') #529
print(f'Total number of params in Model2: {sum(p.numel() for p in Model2_1.parameters())}') #524

#Loss function
loss = torch.nn.MSELoss()

#define learning rate
lr = 0.01

#define epochs
epochs = 3000

#creating empty arrays to store loss values for each input function
loss_values_Model0_1 = []
loss_values_Model1_1 = []
loss_values_Model2_1 = []

loss_values_Model0_2 = []
loss_values_Model1_2 = []
loss_values_Model2_2 = []

for epoch in range(epochs):
    #first function
    #first model
    Model0_1.train()
    y_pred0_1 = Model0_1(x)
    loss0_1 = loss(y_pred0_1, y1)
    loss0_1.backward()
    #back propogating without use of an optimizer as its a simple function
    #wasn't getting good looking graphs so went with this approach
    with torch.no_grad():
        for param in Model0_1.parameters():
            param -= lr * param.grad
    Model0_1.zero_grad()
    loss_values_Model0_1.append(loss0_1.item())

    #second model
    Model1_1.train()
    y_pred1_1 = Model1_1(x)
    loss1_1 = loss(y_pred1_1, y1)
    loss1_1.backward()
    with torch.no_grad():
        for param in Model1_1.parameters():
            param -= lr * param.grad
    Model1_1.zero_grad()
    loss_values_Model1_1.append(loss1_1.item())

    #third model
    Model2_1.train()
    y_pred2_1 = Model2_1(x)
    loss2_1 = loss(y_pred2_1, y1)
    loss2_1.backward()
    with torch.no_grad():
        for param in Model2_1.parameters():
            param -= lr * param.grad
    Model2_1.zero_grad()
    loss_values_Model2_1.append(loss2_1.item())

    #second function
    #first model
    Model0_2.train()
    y_pred0_2 = Model0_2(x)
    loss0_2 = loss(y_pred0_2, y2)
    loss0_2.backward()
    with torch.no_grad():
        for param in Model0_2.parameters():
            param -= lr * param.grad
    Model0_2.zero_grad()
    loss_values_Model0_2.append(loss0_2.item())
    #second model
    Model1_2.train()
    y_pred1_2 = Model1_2(x)
    loss1_2 = loss(y_pred1_2, y2)
    loss1_2.backward()
    with torch.no_grad():
        for param in Model1_2.parameters():
            param -= lr * param.grad
    Model1_2.zero_grad()
    loss_values_Model1_2.append(loss1_2.item())

    #third model
    Model2_2.train()
    y_pred2_2 = Model2_2(x)
    loss2_2 = loss(y_pred2_2, y2)
    loss2_2.backward()
    with torch.no_grad():
        for param in Model2_2.parameters():
            param -= lr * param.grad
    Model2_2.zero_grad()
    loss_values_Model2_2.append(loss2_2.item())

# plot curves
fig, axes = plt.subplots(2,2, figsize = (20, 16))

fig.text(0.5, 0.95, 'Function 1 y = x^3 - x^2')
axes[0, 0].plot(loss_values_Model0_1, label = 'Model0 Loss')
axes[0, 0].plot(loss_values_Model1_1, label = 'Model1 Loss')
axes[0, 0].plot(loss_values_Model2_1, label = 'Model2 Loss')
axes[0, 0].legend()
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 1].plot(x, y_pred0_1.detach().numpy(), label = 'Model0 Output')
axes[0, 1].plot(x, y_pred1_1.detach().numpy(), label = 'Model1 Output')
axes[0, 1].plot(x, y_pred2_1.detach().numpy(), label = 'Model2 Output')
axes[0, 1].plot(x, y1, label = 'y = x^3 - x^2')
axes[0, 1].legend()
axes[0, 1].set_title('Function Output')

fig.text(0.5, 0.48, 'Function 1 y = cos(x)')
axes[1, 0].plot(loss_values_Model0_2, label = 'Model0 Loss')
axes[1, 0].plot(loss_values_Model1_2, label = 'Model1 Loss')
axes[1, 0].plot(loss_values_Model2_2, label = 'Model2 Loss')
axes[1, 0].legend()
axes[1, 0].set_title('Model Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 1].plot(x, y_pred0_2.detach().numpy(), label = 'Model0 Output')
axes[1, 1].plot(x, y_pred1_2.detach().numpy(), label = 'Model1 Output')
axes[1, 1].plot(x, y_pred2_2.detach().numpy(), label = 'Model2 Output')
axes[1, 1].plot(x, y2, label = 'y = cos(x)')
axes[1, 1].legend()
axes[1, 1].set_title('Function Output')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()