import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, d_model, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 100, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(100, 50, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.mean(dim=2)  # Take the mean of the sequence elements
        x = self.fc(x)
        return x
