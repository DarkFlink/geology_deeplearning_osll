import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(8, 50),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(x):
        return self.layers(x)
