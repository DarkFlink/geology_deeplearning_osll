import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np

# constants
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
learn_rate = 0.001
epochs = 15
batch_size = 200

class ClassifierConv(nn.Module):
    def __init__(self):
        super(ClassifierConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(in_features=16 * 16 * 64, out_features=classes.__len__())

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = output.view(-1, 16 * 16 * 64)
        output = self.fc(output)

        return output



# load dataset from lib
transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = CIFAR10(root='./datasets/', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./datasets/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# initializing model
classifier = ClassifierConv()

crossentropy_loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(classifier.parameters(), lr=learn_rate)

print(classifier)

history_test_loss = []
history_train_loss = []
history_test_acc = []
history_train_acc = []

# fit model
# main epochs for
for epoch in range(epochs):
    classifier.train()
    # fitting procedure
    train_loss = 0.0
    for idx, (X, Y) in enumerate(train_loader):
        opt.zero_grad()

        # forward -->
        model_output = classifier.forward(X)

        # classify model output
        loss_output = crossentropy_loss(model_output, Y)
        # backward <--
        loss_output.backward()

        # optimizer need to update weights
        opt.step()

        train_loss += loss_output.cpu().item() * X.size(0)

    # testing
    classifier.eval()
    test_loss = 0.0
    for idx, (X,Y) in enumerate(test_loader):

        model_output = classifier.forward(X)

        loss_output = crossentropy_loss(model_output, Y)

        test_loss += loss_output.cpu().item() * X.size(0)

    # for tracking history
    test_loss = test_loss / 10000
    train_loss = train_loss / 50000
    history_train_loss.append(train_loss)
    history_test_loss.append(test_loss)
    print(f'Epoch: {epoch}, loss: {train_loss:.5f}, val_loss: {test_loss:.5f}')

plt.figure(1, figsize=(10, 10))
plt.title("Loss")
plt.plot(history_test_loss, 'r', label='test')
plt.plot(history_train_loss, 'b', label='train')
plt.legend()
plt.show()
plt.clf()
