import torch
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from model import Net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

raw_data = pandas.read_csv("data/winequality-red.csv")
data, targets = raw_data.drop('quality', axis=1), raw_data['quality']

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.3, random_state=101)

train_data = torch.Tensor(np.array(train_data)).to(device)
train_targets = torch.Tensor(np.array(train_targets)).to(device)
test_data = torch.Tensor(np.array(test_data)).to(device)
test_targets = torch.Tensor(np.array(test_targets)).to(device)

train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=10, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=10, shuffle=True)
