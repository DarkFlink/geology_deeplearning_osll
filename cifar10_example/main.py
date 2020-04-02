import torch
import torchvision.datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from cifar10_example.model import Net

num_epochs = 10
batch_size = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             download=True,
                                             train=True,
                                             transform=transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            download=True,
                                            train=False,
                                            transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(model, train_loader, floss, optimizer, ep_num, train_accuracy_history):
    model.train()
    train_correct = 0.
    total = 0.
    for index, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = net.forward(images)

        loss_eval = floss(outputs, labels)
        loss_eval.backward()
        optimizer.step()

        total += labels.size(0)
        _, train_predicted = outputs.max(1)
        train_correct += train_predicted.eq(labels).sum().item()
        if index != 0 and index % 200 == 0:
            train_acc = train_correct / total
            train_accuracy_history += [train_acc]
            print(f'Epoch {ep_num} - Train acc {train_acc:.3f}')


def test(model, test_loader, ep_num, test_accuracy_history):
    model.eval()
    with torch.no_grad():
        test_correct = 0.
        total = 0.
        for index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net.forward(images)

            total += labels.size(0)
            _, test_predicted = outputs.max(1)
            test_correct += test_predicted.eq(labels).sum().item()

            if index != 0 and index % 40 == 0:
                test_acc = test_correct / total
                test_accuracy_history += [test_acc]
                print(f'Epoch {ep_num} - Test acc {test_acc:.3f}')


net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

loss = torch.nn.CrossEntropyLoss()

test_accuracy_history = []
train_accuracy_history = []

for epoch in range(num_epochs):
    train(net, train_loader, loss, optimizer, epoch, train_accuracy_history)
    test(net, test_loader, epoch, test_accuracy_history)

plt.figure(1, figsize=(10, 8))
plt.plot(test_accuracy_history, color='blue', label='test')
plt.plot(train_accuracy_history, color='red', label='train')
plt.legend(loc='upper left')

torch.save(net.state_dict(), "./model2.pth")
plt.savefig("./accuracy2.png")