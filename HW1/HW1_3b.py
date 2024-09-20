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

#transform for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#create train and test data sets from MNIST
train_dataset = datasets.MNIST(root= './data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root= './data', train=False, transform=transform, download=True)

#create train and test data loaders with batch size of 256
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

#function to create a DNN model with input of layer sizes
def create_dnn(layer_sizes):
    layers = []
    input_size = 28 * 28
    for size in layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(nn.ReLU())
        input_size = size
    layers.append(nn.Linear(input_size, 10))
    return nn.Sequential(*layers)

#function to train and test the model
def train_and_test_model(model):
    epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    #variables to track loss and accuracy from training and testing
    train_loss_values = []
    train_accuracy_values = []
    test_loss_values = []
    test_accuracy_values = []

    #train for number of epochs
    for i in range(epochs):

        #print statement to track which epoch during training
        print(f'epoch {i}')
        model = model.to(device)
        model.train()

        #variables to track loss and correct/total for accuracy for test
        total_loss = 0
        total_correct_train = 0
        total_predicted_train = 0

        #train to train loader
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            #track loss and total correct and total predicted
            total_loss += loss.item()
            total_correct_train += torch.sum(output.argmax(dim=1) == labels).item()
            total_predicted_train += labels.size(0)

        #measure average loss and accuracy over training
        train_loss_values.append(total_loss/len(train_loader))
        train_accuracy_values.append(total_correct_train/total_predicted_train)

        #set model to eval mode
        model.eval()

        #variables to track loss and correct/total for accuracy for train
        total_loss = 0
        total_correct_test = 0
        total_predicted_test = 0

        #test with test loader with no gradient because not needed and speeds it up
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.size(0), -1)
                output = model(images)
                loss = loss_fn(output, labels)

                # track loss and total correct and total predicted
                total_loss += loss.item()
                total_correct_test += torch.sum(output.argmax(dim=1) == labels).item()
                total_predicted_test += labels.size(0)

        # measure average loss and accuracy over test
        test_loss_values.append(total_loss/len(test_loader))
        test_accuracy_values.append(total_correct_test/total_predicted_test)

    #return loss and accuracy values
    return train_loss_values, train_accuracy_values, test_loss_values, test_accuracy_values


#create a list of lists that will create a 2 hidden layer function what that number
#nuerons per layer
layer_configs = [
    [16, 16],
    [32, 32],
    [64, 64],
    [96, 96],
    [128, 128],
    [192, 192],
    [224, 224],
    [256, 256],
    [384, 384],
    [512, 512],
    [768, 768],
    [1024, 1024],
    [1536, 1536],
    [2048, 2048],
    [2560, 2560]
]

#variables to track losses and accuracies and number of parameters
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
num_params = []

#create the different configurations of models from list of configurations
for config in layer_configs:
    #create model based on list
    model = create_dnn(config)
    #count params
    total_params = sum(p.numel() for p in model.parameters())
    num_params.append(total_params)

    #train and test model
    train_loss, train_acc, test_loss, test_acc = train_and_test_model(model)

    #keep track just the final loss and accuracy
    train_losses.append(train_loss[-1])
    test_losses.append(test_loss[-1])
    train_accuracies.append(train_acc[-1])
    test_accuracies.append(test_acc[-1])

#plot the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(num_params, train_losses, label='train_loss', color='blue')
plt.scatter(num_params, test_losses, label='test_loss', color='orange')
plt.xlabel('Number of Parameters')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(num_params, train_accuracies, label='train_acc', color='blue')
plt.scatter(num_params, test_accuracies, label='test_acc', color='orange')
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


