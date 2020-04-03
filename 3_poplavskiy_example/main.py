import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

n_epoch = 30
lr = 10e-4
batch_size = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 250)
        self.fc2 = nn.Linear(250, 115)
        self.fc3 = nn.Linear(115, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

loss_arr = []
for epoch in tqdm_notebook(range(n_epoch)):
    net.train()
    curr_loss = 0.0
    i = 0
    for batch in tqdm_notebook(trainloader):
        optimizer.zero_grad()
        inputs, target = batch
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()
        i+=1
    print('[%d] loss: %.3f' % (epoch + 1, curr_loss /i))
    loss_arr.append(curr_loss/i)
    curr_loss = 0.0
    
        
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Точность сети на 10000 тестовых изображений: {}%".format(100 * correct / total))
plt.plot(range(n_epoch),loss_arr)