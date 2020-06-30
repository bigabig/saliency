import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, 2)

        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        self.fc4 = nn.Linear(64 * 28 * 28, 100)
        self.fc5 = nn.Linear(100, 2)
        self.fc6 = nn.Linear(2, 1)

    def forward(self, xb):
        xb = F.relu(self.conv1_1(xb))
        xb = F.relu(self.conv1_2(xb))
        xb = self.pool1(xb)

        xb = F.relu(self.conv2_1(xb))
        xb = F.relu(self.conv2_2(xb))
        xb = self.pool2(xb)

        xb = F.relu(self.conv3_1(xb))
        xb = F.relu(self.conv3_2(xb))
        xb = self.pool3(xb)

        xb = xb.view(-1, 64*28*28)

        xb = F.tanh(self.fc4(xb))
        xb = F.tanh(self.fc5(xb))

        xb = self.fc6(xb)

        return xb
