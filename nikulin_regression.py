from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

input_size = 8
hidden_size = 32
num_epochs = 300
batch_size = 50
learning_rate = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

x, y = fetch_california_housing(return_X_y=True)
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.float).to(device).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float).to(device).view(-1, 1)

train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

class RegressionNet(nn.Module):
  def __init__(self):
    super(RegressionNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = RegressionNet().to(device)
criterion = nn.MSELoss()
optimizer = opt.SGD(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for x, y in train_loader:
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
  if (epoch + 1) % 10 == 0:
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

with torch.no_grad():
  test_output = net(x_test)
  plt.figure(1, figsize=(20, 10))
  plt.title('Values')
  plt.plot(y_test.cpu().data[0:100], label='Targets')
  plt.plot(test_output.cpu().data[0:100], label='Predictions')
  plt.legend()