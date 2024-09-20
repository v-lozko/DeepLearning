# Valery Lozko
# CPSC 8430 Fall 2024
# HW 1

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#create MNIST train and test data set
data_train = datasets.MNIST(root='./data', train = True, transform = transform, download=True)
data_test = datasets.MNIST(root='./data', train = False, transform = transform, download=True)

#create train loaders of 64 and 256 batch size, create a test loader
train_loader_64 = DataLoader(data_train, batch_size = 64, shuffle = True)
train_loader_256 = DataLoader(data_train, batch_size = 256, shuffle = True)
test_loader = DataLoader(data_test, batch_size = 64, shuffle = True)

#create an alpha variable, array of 10 values between 0 and 1
alpha = torch.linspace(0,1,10)

#define model
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

#function to train model
def train_model(model, train_loader):

    #see where i am in training
    print('training model....')
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    #train for number of epochs, will use train loader passed in
    for epoch in range(40):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    #return model
    return model

#function to create interpolated models
def interpolate_models(model1, model2, alpha):
    interpolated_model = DNN()
    interpolated_model.to(device)
    with torch.no_grad():

        #runs through the parameters sequentially of model 1, model 2, and the interpolated model
        for params1, params2, params_interpolated in zip(model1.parameters(), model2.parameters(), interpolated_model.parameters()):

            #set the paramaters in interpolated model to the calculated one based on alpha and model 1 and 2
            params_interpolated.data = (((1-alpha)*params1) + (alpha*params2))

    #return the interpolated model
    return interpolated_model

#function to measure accuracy and loss
def measure_accuracy_and_loss(model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    #no gradient tracking needed as we have already trained model and testing to training
    #testing data sets
    with torch.no_grad():
        total_loss_train = 0.0
        num_correct_train = 0.0
        num_samples_train = 0.0
        for images, labels in train_loader_64:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss_train += loss.item()
            num_correct_train += (outputs.argmax(dim=1) == labels).sum().item()
            num_samples_train += labels.size(0)
        total_loss_train = total_loss_train / len(train_loader_64)
        accuracy_train = num_correct_train / num_samples_train

        total_loss_test = 0.0
        num_correct_test = 0.0
        num_samples_test = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss_test += loss.item()
            num_correct_test = (outputs.argmax(dim=1) == labels).sum().item()
            num_samples_test += labels.size(0)
        total_loss_test = total_loss_test / len(test_loader)
        accuracy_test = num_correct_test / num_samples_test

    return total_loss_train, accuracy_train, total_loss_test, accuracy_test

#create models with 64 and 256 batch size
Model_1 = train_model(DNN(), train_loader_64)
Model_2 = train_model(DNN(), train_loader_256)

#create a variable to hold the interpolated models
interpolated_models = []

#loop through alpha values and create interpoloated model, appending it to
#interpolated_models
for i in alpha:
    interpolated_models.append(interpolate_models(Model_1, Model_2, i))

#create variables to hold loss and accuracy of interpolated models
int_loss_train = []
int_accuracy_train = []
int_loss_test = []
int_accuracy_test = []

#run through the interpolated models and measure accuracy and loss and append it
#to the lists
for model in interpolated_models:

    #print statement so i can follow process
    print('measuring model accuracy and loss')
    loss_train, acc_train, loss_test, acc_test = measure_accuracy_and_loss(model)
    int_loss_train.append(loss_train)
    int_accuracy_train.append(acc_train)
    int_loss_test.append(loss_test)
    int_accuracy_test.append(acc_test)

#plot data
fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

ax1.plot(alpha, int_loss_train, 'r-', label='Train Loss')
ax1.plot(alpha, int_loss_test, 'r--', label = 'Test Loss')
ax2.plot(alpha, int_accuracy_train, 'b-', label = 'Train Accuracy')
ax2.plot(alpha, int_accuracy_test, 'b--', label = 'Test Accuracy')

ax1.set_xlabel('Alpha')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
fig.legend(loc="upper right")
plt.title('Linear Interpolation between 64 and 256 batch size')
plt.show()



