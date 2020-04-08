import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import relu
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')

wineData = pd.read_csv("wine.csv")

wineData.head()

X = wineData[wineData.columns[0:11]].values
y = wineData['quality']
X = (X-X.std(axis=0))/X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(11, 140)
        self.fc2 = nn.Linear(140, 140)
        self.fc3 = nn.Linear(140, 1)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Regression().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

net.train()
for epoch in tqdm_notebook(range(250)):
    loss_epoch = 0
    for x,y in zip(X_train, y_train):
        x = torch.from_numpy(x).to(device).float()
        y = torch.tensor(y).to(device).float()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch += loss.item()
    print("loss=", loss_epoch/X_train.shape[0])

y_pred_arr = []
loss = 0
for x,y in zip(X_test, y_test):
    x = torch.from_numpy(x).to(device).float()
    y = torch.tensor(y).to(device).float()
    y_pred = net(x)
    loss = criterion(y_pred, y)
    y_pred_arr.append(y_pred.item())
    loss += loss.item()
print("loss on test data:", loss.item()) # loss on test data: 0.327

plt.figure(1, figsize=(16,9))
plt.title('Predictions')
plt.plot(y_pred_arr[0:50], label='predict')
plt.plot(list(y_test[0:50]), label='truth')
plt.legend()
plt.show()
