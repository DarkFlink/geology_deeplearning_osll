import torch
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

import numpy as np

from pyexamples.classifier_exapmle import ClassifierExample


image_dim = 784

classes_dim = 10
epochs = 100
batch_size = 50
learning_rate = 0.001
momentum = 0.5

torch.manual_seed(0)

test_data = datasets.MNIST(root="./datasets", train=False, download=True)
train_data = datasets.MNIST(root="./datasets", train=True, download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = train_data.data.to(device)
train_targets = train_data.targets.to(device)

test_data = test_data.data.to(device)
test_targets = test_data.targets.to(device)

train_data = train_data.float()
test_data = test_data.float()

train_data = train_data.reshape([-1, 28 * 28])
test_data = test_data.reshape([-1, 28 * 28])

loss = torch.nn.CrossEntropyLoss()
model = ClassifierExample(image_dim, classes_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(epochs):
    order = np.random.permutation(len(train_data))

    for index in range(0, len(train_data), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[index:index + batch_size]

        data = train_data[batch_indexes].to(device)
        labels = train_targets[batch_indexes].to(device)

        outputs = model.forward(data)

        loss_value = loss(outputs, labels)
        loss_value.backward()

        optimizer.step()

    train_outputs = model.forward(train_data)
    test_outputs = model.forward(test_data)

    train_accuracy = (train_outputs.argmax(dim=1) == train_targets).float().mean()
    test_accuracy = (test_outputs.argmax(dim=1) == test_targets).float().mean()

    train_acc_history.append(train_accuracy)
    test_acc_history.append(test_accuracy)

    test_loss = loss(test_outputs, test_targets)
    test_loss_history.append(test_loss)

    print(f'Epoch: {epoch} - Loss: {test_loss:.6f}')

torch.save(model.state_dict(), "./mnist_model.pth")

plt.figure(1, figsize=(8, 5))
plt.title("train accuracy")
plt.plot(train_acc_history)
plt.savefig("./train_accuracy_history.png")

plt.figure(2, figsize=(8, 5))
plt.title("test accuracy")
plt.plot(test_acc_history)
plt.savefig("./test_accuracy_history.png")

plt.figure(3, figsize=(8, 5))
plt.title("test loss history")
plt.plot(test_loss_history)
plt.savefig("./test_loss_history.png")



