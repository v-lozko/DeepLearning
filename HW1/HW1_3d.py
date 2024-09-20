# Valery Lozko
# CPSC 8430 Fall 2024
# HW 1

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#training and testing data sets from MNIST
data_train = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
data_test = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

#test data loader
test_loader = DataLoader(data_test, batch_size = 64, shuffle = True)

#batch sizes that will be fed into the generation of the training data loader
batch_sizes = [ 32, 64, 96, 128, 192, 256, 512, 1024]

#define the model
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#function to train the model, feed in the batch size
def train_model(model, batch_size):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    model.train()

    #create a data loader for training data set based on passed in batch size
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)

    #train for 40 epochs
    for i in range(40):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    #return the model
    return model

#function to get loss, accuracy, and sensitivity of the model
def loss_acc_sens(model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(data_train, batch_size = 64, shuffle = True)

    total_loss_train = 0
    num_correct_train = 0
    num_examples_train = 0
    fro_norm_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss_train += loss.item()
        num_correct_train += (outputs.argmax(dim=1) == labels).sum().item()
        num_examples_train += labels.size(0)

        model.zero_grad()
        loss.backward()

        #calculuate and sum up the frobius norm over the training data set
        for p in model.parameters():
            fro_norm_train += torch.norm(p.grad, 'fro')

    #average out the frobius norm based on length of train loader
    fro_norm_train = fro_norm_train/len(train_loader)
    total_loss_train = total_loss_train / len(train_loader)
    accuracy_train = num_correct_train / num_examples_train

    total_loss_test = 0
    num_correct_test = 0
    num_examples_test = 0
    fro_norm_test = 0

    #do the same thing for the testing data with test loader
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss_test += loss.item()
        num_correct_test += (outputs.argmax(dim=1) == labels).sum().item()
        num_examples_test += labels.size(0)

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            fro_norm_test += torch.norm(p.grad, 'fro')

    total_loss_test = total_loss_test / len(test_loader)
    accuracy_test = num_correct_test / num_examples_test
    fro_norm_test = fro_norm_test/len(test_loader)

    #return loss, accuracy, and sensitivity (called fro_norm here)
    return total_loss_train, accuracy_train, fro_norm_train, total_loss_test, accuracy_test, fro_norm_test

#create variable to hold models
trained_models = []

#train the models of varying batch sizes and append model to variable
for batch_size in batch_sizes:
    #so i know how far along i am
    print(f'training model with batch size {batch_size}')
    model = DNN()
    trained_models.append(train_model(model, batch_size))

#variables to hold results
loss_train = []
acc_train = []
sensitivity_train = []
loss_test = []
acc_test = []
sensitivity_test = []

#loop through models and append the results to the initialized variables
for trained_model in trained_models:
    train_loss, train_acc, sen_train, test_loss, test_acc, sen_test = loss_acc_sens(trained_model)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    sensitivity_train.append(sen_train.cpu().numpy())
    loss_test.append(test_loss)
    acc_test.append(test_acc)
    sensitivity_test.append(sen_test.cpu().numpy())

#plot it
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

ax1.plot(batch_sizes, loss_train, 'b-', label='Train loss')
ax1.plot(batch_sizes, loss_test, 'b--', label='Test loss' )
ax2.plot(batch_sizes, sensitivity_train, 'r-', label='Train sensitivity')
ax2.plot(batch_sizes, sensitivity_test, 'r--', label='Test sensitivity')
ax1.set_xscale('log')
ax1.set_xlabel('Batch Size (log scale')
ax1.set_ylabel('Loss')
ax2.set_label('Sensitivity')

fig1.legend(loc="upper right")
plt.title('Loss and Sensitivity vs. Batch Size')
plt.show()

fig2, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

ax1.plot(batch_sizes, acc_train, 'b-', label='Train accuracy')
ax1.plot(batch_sizes, acc_test, 'b--', label='Test accuracy' )
ax2.plot(batch_sizes, sensitivity_train, 'r-', label='Train sensitivity')
ax2.plot(batch_sizes, sensitivity_test, 'r--', label='Test sensitivity')
ax1.set_xscale('log')
ax1.set_xlabel('Batch Size (log scale')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Sensitivity')

fig2.legend(loc="upper right")
plt.title('Accuracy and Sensitivity vs. Batch Size')
plt.show()

