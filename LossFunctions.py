import torch


# Use for binary classification!
class BCELossWithDownsampling:
    def __init__(self):
        self.downsample = torch.nn.AvgPool2d(4, stride=4, count_include_pad=False)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def __call__(self, pred, y):
        return self.loss_fcn(self.downsample(pred), self.downsample(y))


class WeightedMSELoss:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, pred, y):
        prediction = pred
        target = y
        out = ((prediction - target) / (self.alpha - target)) ** 2
        return out.mean()


class WeightedMSELossWithDownsampling:
    def __init__(self, alpha):
        self.alpha = alpha
        self.downsample = torch.nn.AvgPool2d(4, stride=4, count_include_pad=False)

    def __call__(self, pred, y):
        prediction = self.downsample(pred)
        target = self.downsample(y)
        out = ((prediction - target) / (self.alpha - target)) ** 2
        return out.mean()
