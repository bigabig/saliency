import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc4 = nn.Linear(64 * 28 * 28, 100)
        self.fc5 = nn.Linear(100, 2)
        self.fc6 = nn.Linear(2, 1)

    def forward(self, xb):
        # 4 channel input: r,g,b from ground truth, s from predicted or ground truth saliency map
        xb = xb.view(-1, 4, 224, 224)  # Images as input: nSamples x nChannels x Height x Width

        xb = F.relu(self.conv1_1(xb))
        xb = F.relu(self.conv1_2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv2_1(xb))
        xb = F.relu(self.conv2_2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv3_1(xb))
        xb = F.relu(self.conv3_2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = xb.view(-1, 64*28*28)

        xb = F.tanh(self.fc4(xb))
        xb = F.tanh(self.fc5(xb))

        xb = self.fc6(xb)
        # xb = F.sigmoid(self.fc6(xb))

        return xb
