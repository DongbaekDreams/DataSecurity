'''Shared MNIST models used by Levels 1–3.

LeNetWide32 is the primary model (widened LeNet on a 32x32 canvas).
LeNet5 is the classic baseline for comparison.
'''

from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    '''Classic LeNet-5 widths, ReLU, average pooling.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.avg_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.avg_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


class LeNetWide32(nn.Module):
    '''Widened LeNet on the original 32x32 canvas, max-pool. Primary model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.pad(xb, (2, 2, 2, 2))
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


def make_model(name="wide"):
    name = name.lower()
    if name in ("wide", "lenetwide32", "primary"):
        return LeNetWide32()
    if name in ("lenet5", "lenet", "baseline"):
        return LeNet5()
    raise ValueError(f"Unknown model {name!r}; use 'wide' or 'lenet5'")
