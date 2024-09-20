# Valery Lozko
# CPSC 8430 Fall 2024
# HW 1

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#define x-values and function
x = torch.linspace(-1, 1, 1000).reshape(-1, 1).to(device)
y = (x - x**2).to(device)

#create a function to compute the hessian
def compute_hessian(output, model_params):
    hessian = []
    #compute gradient as iterating over parameters
    for param in model_params:
        #first order derivitive calculation
        grads = torch.autograd.grad(output, param, create_graph=True, retain_graph=True)[0]
        grads = grads.view(-1)  # Flatten to make sure we handle each parameter element-wise
        hess_param = []
        #calculate second order derivitive
        for grad in grads:
            hess_row = torch.autograd.grad(grad, param, create_graph=True)[0].view(-1)
            hess_param.append(hess_row)

        #append the hessian matrix
        hessian.append(torch.stack(hess_param))

    #return hessian
    return hessian

#function to compute the gradient norm
def compute_gradient_norm(model, loss):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grad_norm = 0
    for grad in grads:
        grad_norm += torch.sum(grad ** 2)
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

#function to train and compute minimal ratio
def train_and_compute_minimal_ratio(n_iterations=100):
    model = DNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    #variables to track minimal ratios and loss
    minimal_ratios = []
    loss_values = []

    for iteration in range(n_iterations):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        #compute the gradient norm
        grad_norm = compute_gradient_norm(model, loss)

        #check if norm is low
        if grad_norm.cpu().detach().numpy() < 0.01:
            print('low gradient norm')
            grad_norm.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_values.append(loss.item())

        #compute the hessian
        hessian = compute_hessian(loss, model.parameters())

        #compute eigen values for each hessian
        total_positive_eigenvalues = 0
        total_eigenvalues = 0
        for param_hessian in hessian:
            eigenvalues, _ = torch.linalg.eig(param_hessian)
            eigenvalues = eigenvalues.real

            #measure how many positive and how many total eigen values
            positive_eigenvalues = eigenvalues[eigenvalues > 0]
            total_positive_eigenvalues += len(positive_eigenvalues)
            total_eigenvalues += len(eigenvalues)

        #calculate minimal ratio
        minimal_ratio = total_positive_eigenvalues / total_eigenvalues if total_eigenvalues > 0 else 0
        minimal_ratios.append(minimal_ratio)

        #have print statement to track and see loss and minimal ratio as data doesn't look right
        print(f"Iteration {iteration + 1}, Loss: {loss.item()}, Minimal Ratio: {minimal_ratio}")

    #return minimal ratios and loss
    return minimal_ratios, loss_values

#run function to train and compute minimal ration (and loss)
minimal_ratios, losses = train_and_compute_minimal_ratio(n_iterations=100)

#plot data
plt.figure(figsize=(10, 5))
plt.scatter(minimal_ratios, losses)
plt.xlabel("Minimal Ratio")
plt.ylabel("Loss")
plt.title("Minimal Ratio vs Loss over 100 Training Iterations")
plt.show()
