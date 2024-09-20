# Valery Lozko
# CPSC 8430 Fall 2024
# HW 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#create data sets from training and test from MNIST
train_dataset = datasets.MNIST(root= './data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root= './data', train=False, transform=transform, download=True)

#shuffle the labels on the data set
shuffled_labels = torch.randperm(len(train_dataset.targets))
train_dataset.targets = train_dataset.targets[shuffled_labels]

#make train (after shuffle) and test data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

#define function
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#function to train and test model
def train_and_test_model(model):
    epochs = 1000
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #variables to store loss values of training and testing
    train_loss_values = []
    test_loss_values = []

    #train for epochs
    for i in range(epochs):
        model = model.to(device)
        model.train()

        #variable to track total loss
        total_loss = 0

        #run through data in train loader to train model
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #keep track of average loss over training
        train_loss_values.append(total_loss/len(train_loader))

        model.eval()
        total_loss = 0
        #no grad as we are evaluating and dont need to track gradients
        with torch.no_grad():
            #run through test loader data and train model
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.size(0), -1)
                output = model(images)
                loss = loss_fn(output, labels)
                total_loss += loss.item()

        #keep track of average loss over testing
        test_loss_values.append(total_loss/len(test_loader))

        #print statement to track where it is in process
        print(f'epoch {i}')

    #return loss values from training and testing
    return train_loss_values, test_loss_values

#train and test model and get loss values
train_loss, test_loss = train_and_test_model(DNN())

#plot results
plt.figure(figsize=(10, 5))

plt.scatter(range(len(train_loss)), train_loss, label='train_loss', color='blue')
plt.scatter(range(len(test_loss)), test_loss, label='train_loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()


