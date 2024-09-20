#Valery Lozko
#CPSC 8430 Fall 2024
#HW 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform for the MNIST data set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#create the training and testing data sets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#create data loaders with batch size 128
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#define CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*7*7, 64)
        self.fc2 = nn.Linear(64, 10)  #MNIST has 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#define DNN model
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#define epochs
epochs = 30

#function to train and test model
def train_test_model(NN_model, epochs):
    #create model
    model = NN_model.to(device)
    #define loss function
    loss_fn = nn.CrossEntropyLoss()
    #define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #create variables to store loss and accuracy
    loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    #trainin loop for number of epochs
    for epoch in range(epochs):
        model.train()

        #define variables needed to calculate loss and accuracy
        total_loss = 0
        total_correct_train = 0
        total_predicted_train = 0
        total_correct_test = 0
        total_predicted_test = 0

        #train model with batch sizes
        for images, labels in train_loader:
            images, labels = images, labels
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct_train += torch.sum(output.argmax(dim=1) == labels).item()
            total_predicted_train += labels.size(0)

        #calculate average loss for epoch
        loss_values.append(total_loss / len(train_loader))

        #caculate accuracy for training data
        train_accuracy_values.append(total_correct_train/total_predicted_train)

        #evaluate model
        model.eval()
        #no grad needed as we arent calculating it
        with torch.no_grad():
            #evaluate with test loader batches
            for images, labels in test_loader:
                images, labels = images, labels
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                total_correct_test += torch.sum(output.argmax(dim=1) == labels).item()
                total_predicted_test += labels.size(0)

        # caculate accuracy against test data
        test_accuracy_values.append(total_correct_test/total_predicted_test)

    return loss_values, train_accuracy_values, test_accuracy_values

#get loss and accuracy by running the train and test function
CNN_loss, CNN_train_accuracy, CNN_test_accuracy = train_test_model(CNN(), epochs)
DNN_loss, DNN_train_accuracy, DNN_test_accuracy = train_test_model(DNN(), epochs)

#plot data
fig1, axes = plt.subplots(1,2, figsize = (16, 5))
fig1.suptitle('Loss During Training of Models')
axes[0].plot(range(epochs), CNN_loss)
axes[0].set_title('CNN Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[1].plot(range(epochs), DNN_loss)
axes[1].set_title('DNN Training Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

fig2, axes = plt.subplots(1,2, figsize = (16, 5))
fig2.suptitle('Accuracy of Models')
axes[0].plot(range(epochs), CNN_train_accuracy, label = 'CNN Training Accuracy')
axes[0].plot(range(epochs), CNN_test_accuracy, label = 'CNN Testing Accuracy')
axes[0].legend()
axes[0].set_title('CNN Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[1].plot(range(epochs), DNN_train_accuracy, label = 'DNN Training Accuracy')
axes[1].plot(range(epochs), DNN_test_accuracy, label = 'DNN Testing Accuracy')
axes[1].legend()
axes[1].set_title('DNN Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')

plt.show()



