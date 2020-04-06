import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./datasets/monthly_temperature_aomori_city.csv")
df.head()

Y = df['temperature']
Y.head()

X = df.drop('temperature', axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1488)
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)

X_train[:, 0] = X_train[:, 0] / max(X_train[:, 0])
X_train[:, 1] = X_train[:, 1] / max(X_train[:, 1])

X_test[:, 0] = X_test[:, 0] / max(X_test[:, 0])
X_test[:, 1] = X_test[:, 1] / max(X_test[:, 1])

model = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(300):

    losses = []
    for i in range(X_train.shape[0]):
        x = torch.tensor(X_train[i]).unsqueeze(0).float()
        y = torch.tensor(y_train[i]).unsqueeze(0)

        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    if (epoch + 1) % 4 == 0:
        print(np.mean(losses))

        with torch.no_grad():
            y_s = []
            p_s = []
            for i in range(1):
                for i in range(X_train.shape[0]):
                    x = torch.tensor(X_train[i]).unsqueeze(0).float()
                    y = torch.tensor(y_train[i]).unsqueeze(0)

                    preds = model(x)

                    y_s.append(y.item())
                    p_s.append(preds.item())

            plt.figure(1, figsize=(14,6))
            plt.title('Predictions')
            plt.plot(y_s[0:100], label='truth')
            plt.plot(p_s[0:100], label='pred')
            plt.legend()
            plt.show()
