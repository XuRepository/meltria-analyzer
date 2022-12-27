from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_TO_CATEGORY = OrderedDict(
    [
        (0, "Fluctuations/anomaly_during_fault"),
        (1, "Fluctuations/anomaly_outside_fault"),
        (2, "Level shift down/anomaly_during_fault"),
        (3, "Level shift down/anomaly_outside_fault"),
        (4, "Level shift up/anomaly_during_fault"),
        (5, "Level shift up/anomaly_outside_fault"),
        (6, "Multiple dips/anomaly_during_fault"),
        (7, "Multiple dips/anomaly_outside_fault"),
        (8, "Multiple spikes/anomaly_during_fault"),
        (9, "Multiple spikes/anomaly_outside_fault"),
        (10, "Other normal/no_anomaly"),
        (11, "Single dip/anomaly_during_fault"),
        (12, "Single dip/anomaly_outside_fault"),
        (13, "Single spike/anomaly_during_fault"),
        (14, "Single spike/anomaly_outside_fault"),
        (15, "Steady decrease/anomaly_during_fault"),
        (16, "Steady decrease/anomaly_outside_fault"),
        (17, "Steady increase/anomaly_during_fault"),
        (18, "Steady increase/anomaly_outside_fault"),
        (19, "Sudden decrease/anomaly_during_fault"),
        (20, "Sudden decrease/anomaly_outside_fault"),
        (21, "Sudden increase/anomaly_during_fault"),
        (22, "Sudden increase/anomaly_outside_fault"),
        (23, "Transient level shift down/anomaly_during_fault"),
        (24, "Transient level shift down/anomaly_outside_fault"),
        (25, "Transient level shift up/anomaly_during_fault"),
        (26, "Transient level shift up/anomaly_outside_fault"),
        (27, "White noise/no_anomaly"),
    ]
)

NORMAL_CLASSES: set[int] = set([10, 27])
ANONALY_CLASSES: set[int] = set([i for i in CLASS_TO_CATEGORY.keys() if i not in NORMAL_CLASSES])

TYPE0_CLASSES: set[int] = NORMAL_CLASSES
TYPE1_CLASSES: set[int] = set([2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22])
TYPE2_CLASSES: set[int] = set([i for i in CLASS_TO_CATEGORY.keys() if i not in TYPE0_CLASSES.union(TYPE1_CLASSES)])


CLASS_TO_CATEGORY_WITHOUT_AP = OrderedDict(
    [
        (0, "Fluctuations"),
        (1, "Level shift down"),
        (2, "Level shift up"),
        (3, "Multiple dips"),
        (4, "Multiple spikes"),
        (5, "Other normal"),
        (6, "Single dip"),
        (7, "Single spike"),
        (8, "Steady decrease"),
        (9, "Steady increase"),
        (10, "Sudden decrease"),
        (11, "Sudden increase"),
        (12, "Transient level shift down"),
        (13, "Transient level shift up"),
        (14, "White noise"),
    ]
)

NORMAL_CLASSES_WITHOUT_AP: set[int] = set([5, 14])
ANONALY_CLASSES_WITHOUT_AP: set[int] = set(
    [i for i in CLASS_TO_CATEGORY_WITHOUT_AP.keys() if i not in NORMAL_CLASSES_WITHOUT_AP]
)

TYPE0_CLASSES_WITHOUT_AP: set[int] = NORMAL_CLASSES_WITHOUT_AP
TYPE1_CLASSES_WITHOUT_AP: set[int] = set([1, 2, 8, 9, 10, 11])
TYPE2_CLASSES_WITHOUT_AP: set[int] = set(
    [
        i
        for i in CLASS_TO_CATEGORY_WITHOUT_AP.keys()
        if i not in TYPE0_CLASSES_WITHOUT_AP.union(TYPE1_CLASSES_WITHOUT_AP)
    ]
)


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
