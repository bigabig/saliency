import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, model_dict=None):
        super(Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        if model_dict is not None:
            self.conv1_1.weight.data = model_dict['conv1_1_weight']
            self.conv1_1.bias.data = model_dict['conv1_1_bias']
            self.conv1_2.weight.data = model_dict['conv1_2_weight']
            self.conv1_2.bias.data = model_dict['conv1_2_bias']

            self.conv2_1.weight.data = model_dict['conv2_1_weight']
            self.conv2_1.bias.data = model_dict['conv2_1_bias']
            self.conv2_2.weight.data = model_dict['conv2_2_weight']
            self.conv2_2.bias.data = model_dict['conv2_2_bias']

            self.conv3_1.weight.data = model_dict['conv3_1_weight']
            self.conv3_1.bias.data = model_dict['conv3_1_bias']
            self.conv3_2.weight.data = model_dict['conv3_2_weight']
            self.conv3_2.bias.data = model_dict['conv3_2_bias']
            self.conv3_3.weight.data = model_dict['conv3_3_weight']
            self.conv3_3.bias.data = model_dict['conv3_3_bias']

            self.conv4_1.weight.data = model_dict['conv4_1_weight']
            self.conv4_1.bias.data = model_dict['conv4_1_bias']
            self.conv4_2.weight.data = model_dict['conv4_2_weight']
            self.conv4_2.bias.data = model_dict['conv4_2_bias']
            self.conv4_3.weight.data = model_dict['conv4_3_weight']
            self.conv4_3.bias.data = model_dict['conv4_3_bias']

            self.conv5_1.weight.data = model_dict['conv5_1_weight']
            self.conv5_1.bias.data = model_dict['conv5_1_bias']
            self.conv5_2.weight.data = model_dict['conv5_2_weight']
            self.conv5_2.bias.data = model_dict['conv5_2_bias']
            self.conv5_3.weight.data = model_dict['conv5_3_weight']
            self.conv5_3.bias.data = model_dict['conv5_3_bias']

    def forward(self, xb):
        xb = xb.view(-1, 3, 224, 224)  # Images as input: nSamples x nChannels x Height x Width

        xb = F.relu(self.conv1_1(xb))
        xb = F.relu(self.conv1_2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv2_1(xb))
        xb = F.relu(self.conv2_2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv3_1(xb))
        xb = F.relu(self.conv3_2(xb))
        xb = F.relu(self.conv3_3(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv4_1(xb))
        xb = F.relu(self.conv4_2(xb))
        xb = F.relu(self.conv4_3(xb))
        xb = F.max_pool2d(xb, kernel_size=2, padding=0, stride=2)

        xb = F.relu(self.conv5_1(xb))
        xb = F.relu(self.conv5_2(xb))
        xb = F.relu(self.conv5_3(xb))

        return xb
