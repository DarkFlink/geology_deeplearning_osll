import torch
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt

from mnist_example.model import MNISTNet

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 0.0001

np.random.seed(0)

train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = train_dataset.data.to(device)
train_labels = train_dataset.targets.to(device)

test_data = test_dataset.data.to(device)
test_labels = test_dataset.targets.to(device)

train_data = train_data.float()
test_data = test_data.float()

train_data = train_data.reshape([-1, 28 * 28])
test_data = test_data.reshape([-1, 28 * 28])

mnist_net = MNISTNet(input_size, hidden_size, num_classes).to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=learning_rate)

train_accuracy_history = []
test_accuracy_history = []
test_loss_history = []

for epoch in range(num_epochs):
    order = np.random.permutation(len(train_data))

    for index in range(0, len(train_data), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[index:index + batch_size]

        data = train_data[batch_indexes].to(device)
        labels = train_labels[batch_indexes].to(device)

        outputs = mnist_net.forward(data)

        loss_value = loss(outputs, labels)
        loss_value.backward()

        optimizer.step()

    train_outputs = mnist_net.forward(train_data)
    test_outputs = mnist_net.forward(test_data)

    train_accuracy = (train_outputs.argmax(dim=1) == train_labels).float().mean()
    test_accuracy = (test_outputs.argmax(dim=1) == test_labels).float().mean()

    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)

    test_loss = loss(test_outputs, test_labels)
    test_loss_history.append(test_loss)

    print(f'Epoch: {epoch} - Loss: {test_loss:.6f}')

torch.save(mnist_net.state_dict(), "./mnist_model.pth")

plt.figure(1, figsize=(8, 5))
plt.title("train accuracy")
plt.plot(train_accuracy_history)
plt.savefig("./train_accuracy_history.png")

plt.figure(2, figsize=(8, 5))
plt.title("test accuracy")
plt.plot(test_accuracy_history)
plt.savefig("./test_accuracy_history.png")

plt.figure(3, figsize=(8, 5))
plt.title("test loss history")
plt.plot(test_loss_history)
plt.savefig("./test_loss_history.png")