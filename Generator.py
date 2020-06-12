import torch.nn as nn

from Decoder import Decoder
from Encoder import Encoder


class Generator(nn.Module):
    def __init__(self, model_dict=None):
        super(Generator, self).__init__()
        self.encoder = Encoder(model_dict)
        self.decoder = Decoder()

    def forward(self, xb):
        xb = self.encoder(xb)
        xb = self.decoder(xb)
        return xb
