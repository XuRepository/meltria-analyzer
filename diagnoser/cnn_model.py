import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN1d, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4608, 64)  # the number of datapoints in a metric = 60/15 * 45 (45min * 15sec interval)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(
            64, num_classes
        )  # The number of class is 13 (chaos types) * 2 (anomaly position) + 2 (normal and unknown)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4, stride=2)
        x = torch.flatten(x, 1)  # Is this necessary?
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, F.softmax(x, dim=1)
