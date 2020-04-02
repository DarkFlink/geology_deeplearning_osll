import torch
import torchvision
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import matplotlib.pyplot as plt

num_epochs = 5
kernel_size = 3
pool_size = 2
conv_1 = 32
conv_2 = 64
conv_3 = 128
hidden_1 = 512
hidden_2 = 256
classes = 10
batch_size = 100
learning_rate = 0.001
final_size = 4

transform = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

class Conv_Net(nn.Module):
  def __init__(self):
    super(Conv_Net, self).__init__()
    self.conv_1 = nn.Conv2d(3, conv_1, kernel_size=kernel_size, padding=1)
    self.bn_1 = nn.BatchNorm2d(conv_1)
    self.conv_2 = nn.Conv2d(conv_1, conv_2, kernel_size=kernel_size, padding=1)
    self.bn_2 = nn.BatchNorm2d(conv_2)
    self.conv_3 = nn.Conv2d(conv_2, conv_3, kernel_size=kernel_size, padding=1)
    self.bn_3 = nn.BatchNorm2d(conv_3)
    self.pool = nn.MaxPool2d(pool_size)
    self.dropout = nn.Dropout(p=0.2)
    self.fc1 = nn.Linear(conv_3 * final_size * final_size, hidden_1)
    self.bn_4 = nn.BatchNorm1d(hidden_1)
    self.fc2 = nn.Linear(hidden_1, hidden_2)
    self.bn_5 = nn.BatchNorm1d(hidden_2)
    self.fc3 = nn.Linear(hidden_2, classes)

  def forward(self, x):
    x = self.pool(F.relu(self.bn_1(self.conv_1(x))))
    x = self.dropout(x)
    x = self.pool(F.relu(self.bn_2(self.conv_2(x))))
    x = self.dropout(x)
    x = self.pool(F.relu(self.bn_3(self.conv_3(x))))
    x = self.dropout(x)
    x = x.view(-1, conv_3 * final_size * final_size)
    x = F.relu(self.bn_4(self.fc1(x)))
    x = self.dropout(x)
    x = F.relu(self.bn_5(self.fc2(x)))
    x = self.dropout(x)
    x = self.fc3(x)
    return x

net = Conv_Net()
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(net.parameters(), lr=learning_rate)
train_acc_plt = []
test_acc_plt = []
total_step = len(train_loader)

for epoch in range(num_epochs):
  for i, (inputs, labels) in enumerate(train_loader, 0):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    if (i + 1) % 100 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                (correct / total) * 100))

  correct = 0
  total = 0
  for data in train_loader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  train_acc_plt.append(correct / total)
  print('Epoch [{}/{}], Train accuracy: {:.2f}%'
    .format(epoch + 1, num_epochs, (correct / total) * 100))

  correct = 0
  total = 0
  for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  test_acc_plt.append(correct / total)
  print('Epoch [{}/{}], Test accuracy: {:.2f}%'
    .format(epoch + 1, num_epochs, (correct / total) * 100))

plt.figure()
plt.title("Accuracy")
plt.plot(train_acc_plt, label='Train')
plt.plot(test_acc_plt, label='Test')
plt.legend()

correct = 0
total = 0
net.eval()
with torch.no_grad():
  for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%' % (100 * correct / total))