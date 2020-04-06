import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from model import Net
from sklearn.model_selection import train_test_split

num_epochs = 5000

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

raw_data = pandas.read_csv("data/housing.csv")
raw_data.columns = raw_data.columns.str.strip() \
    .str.lower() \
    .str.replace(' ', '_') \
    .str.replace('-', '_') \
    .str.replace('\n', '_')
raw_data.describe()

data = raw_data.drop(['ptratio', 'medv'], axis=1)
data = data.apply(lambda x: (x - x.mean()) / x.std())

targets = raw_data['medv']
targets = (targets - targets.mean()) / targets.std()

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2)

train_data = torch.tensor(np.array(data), dtype=torch.float).to(device)
train_targets = torch.tensor(np.array(targets), dtype=torch.float).to(device).view(-1, 1)

test_data = torch.tensor(np.array(data), dtype=torch.float).to(device)
test_targets = torch.tensor(np.array(targets), dtype=torch.float).to(device).view(-1, 1)


def train(model, train, floss, optimizer, ep_num, loss_history):
    model.train()
    data, labels = train
    optimizer.zero_grad()
    outputs = net.forward(data)

    loss = floss(outputs, labels)

    loss.backward()
    optimizer.step()

    if ep_num % 100 == 0:
        loss_history += [loss.item()]
        print(f'Epoch {ep_num} - Loss val {loss.item() :.3f}')


net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

loss = torch.nn.MSELoss()

loss_history = []

for epoch in range(num_epochs):
    train(net, (train_data, train_targets), loss, optimizer, epoch, loss_history)

net.eval()
with torch.no_grad():
    outputs = net.forward(test_data)

plt.figure(1, figsize=(10, 8))
plt.plot(loss_history, color='green', label='loss')
plt.legend(loc='upper right')
plt.plot()
plt.savefig('loss_history.png')

plt.figure(2, figsize=(20, 8))
plt.plot(test_targets, marker='o', color='blue', label='labels')
plt.plot(outputs, marker='x', color='red', label='preds')
plt.legend(loc='upper left')
plt.plot()

plt.savefig('regression.png')
