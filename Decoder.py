import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv5_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # self.output = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0) # TODO: CHANGE
        self.output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, xb):
        xb = F.relu(self.conv1_1(xb))
        xb = F.relu(self.conv1_2(xb))
        xb = F.relu(self.conv1_3(xb))
        xb = self.upsample1(xb)

        xb = F.relu(self.conv2_1(xb))
        xb = F.relu(self.conv2_2(xb))
        xb = F.relu(self.conv2_3(xb))
        xb = self.upsample2(xb)

        xb = F.relu(self.conv3_1(xb))
        xb = F.relu(self.conv3_2(xb))
        xb = F.relu(self.conv3_3(xb))
        xb = self.upsample3(xb)

        xb = F.relu(self.conv4_1(xb))
        xb = F.relu(self.conv4_2(xb))
        xb = self.upsample4(xb)

        xb = F.relu(self.conv5_1(xb))
        xb = F.relu(self.conv5_2(xb))

        xb = self.output(xb)

        return xb
