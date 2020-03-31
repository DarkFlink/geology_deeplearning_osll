import torch


class MNISTNet(torch.nn.Module):
    def __init__(self, input_size, n_hidden_neurons, output_size):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x