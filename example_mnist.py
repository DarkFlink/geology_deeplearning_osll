import torch
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR

import numpy as np

from pyexamples.classifier_exapmle import ClassifierExample


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


image_dim = 1400
classes_dim = 10
epochs = 100
batch_size = 50
learning_rate = 0.001
momentum = 0.5

torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_loader = DataLoader.DataLoader(datasets.MNIST(root="./datasets", train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)
train_loader = DataLoader.DataLoader(datasets.MNIST(root="./datasets", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

#test_data = datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
#train_data = datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train_data = train_data.data.to(device)
#train_targets = train_data.targets.to(device)

#test_data = test_data.data.to(device)
#test_targets = test_data.targets.to(device)

model = ClassifierExample(image_dim, classes_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

scheduler = StepLR(optimizer, step_size=1)
for epoch in range(1, epochs + 1):
    train(10, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
