import torch.nn as nn
import torch.nn.functional as F

layer_dim = 456

class ClassifierExample(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ClassifierExample, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x)