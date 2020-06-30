import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv10_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, xb):
        xb = F.relu(self.conv6_1(xb))
        xb = F.relu(self.conv6_2(xb))
        xb = F.relu(self.conv6_3(xb))
        xb = self.upsample1(xb)

        xb = F.relu(self.conv7_1(xb))
        xb = F.relu(self.conv7_2(xb))
        xb = F.relu(self.conv7_3(xb))
        xb = self.upsample2(xb)

        xb = F.relu(self.conv8_1(xb))
        xb = F.relu(self.conv8_2(xb))
        xb = F.relu(self.conv8_3(xb))
        xb = self.upsample3(xb)

        xb = F.relu(self.conv9_1(xb))
        xb = F.relu(self.conv9_2(xb))
        xb = self.upsample4(xb)

        xb = F.relu(self.conv10_1(xb))
        xb = F.relu(self.conv10_2(xb))
        xb = self.conv10_3(xb)

        return xb
