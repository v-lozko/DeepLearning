#Valery Lozko
#CPSC 8430 Fall 2024
#HW 1

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define DNN model
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#create the train set from MNIST and create data loader of batch size 256
train_dataset = datasets.MNIST(root = './data', train = True, transform = transform, download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = 256, shuffle = True)

#train model function
def train_model(model, train = 8):
    #create model
    model = model.to(device)
    #define loss function
    loss_fn = nn.CrossEntropyLoss()
    #define optimizer
    optimizer = optim.Adam(model.parameters(), lr = .001)

    #create variables to stor weights and loss and gradients
    all_collected_weights = []
    all_first_layer_weights = []
    loss_values = []
    gradients = []

    #training loops
    for i in range(train):
        #because this takes a while and i want to track i created print statement
        print(f'training {i}')
        #loop for epochs of each training
        for epoch in range(75):
            #variables to keep track of gradients and weights
            grad_all = 0.0
            collected_weights = []
            #run through data in loader to train model
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                #calculate the L2 norm
                for p in model.parameters():
                    grad = 0.0
                    if p.grad is not None:
                        grad = (p.grad.cpu().data.numpy()**2).sum()
                    grad_all += grad
                grad_norm = grad_all**0.5
                gradients.append(grad_norm)

            #every third epoch collect weights of first layer and whole model
            if (epoch + 1) % 3 == 0:
                #first layer weights
                first_layer = getattr(model, 'fc1')
                first_layer_weights = first_layer.weight.data.cpu().numpy().flatten()
                all_first_layer_weights.append(first_layer_weights)
                #whole model weights
                for p in model.parameters():
                    collected_weights.append(p.data.cpu().numpy().flatten())
                collected_weights = np.concatenate(collected_weights)
                all_collected_weights.append(collected_weights)

    #return weights, loss, and gradients
    return all_collected_weights, all_first_layer_weights, loss_values, gradients

#train model
all_collected_weights, all_first_layer_weights, loss_values, gradients = train_model(DNN())

#function to reduce dimensions with PCA to 2
def pca(weights):
    weights_matrix = np.concatenate([np.array(session).reshape(1, -1) for session in weights], axis=0)
    pca_ = PCA(n_components=2)
    reduced_weights = pca_.fit_transform(weights_matrix)
    return reduced_weights

#create the reduced matrix
reduced_matrix_first_layer = pca(all_first_layer_weights)
reduced_matrix_all = pca(all_collected_weights)

#plot results
fig, axes1 = plt.subplots(1, 2, figsize=(16, 5))
axes1[0].scatter(reduced_matrix_first_layer[:, 0], reduced_matrix_first_layer[:, 1], marker = 'x')
axes1[0].set_title('PCA of First Layer Weights')
axes1[1].scatter(reduced_matrix_all[:, 0], reduced_matrix_all[:, 1], marker = 'x')
axes1[1].set_title('PCA of Full Model Weights')

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 16))
axes2[0].plot(gradients, label="Gradient Norm")
axes2[0].set_title("Gradient Norm over Iterations")
axes2[0].set_xlabel("Iteration")
axes2[0].set_ylabel("Gradient Norm")
axes2[0].legend()

axes2[1].plot(loss_values, label="Loss")
axes2[1].set_title("Loss over Iterations")
axes2[1].set_xlabel("Iteration")
axes2[1].set_ylabel("Loss")
axes2[1].legend()

plt.show()
