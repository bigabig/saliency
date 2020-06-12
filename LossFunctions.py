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
        self.downsample = torch.nn.AvgPool2d(4, stride=4, count_include_pad=False)
        self.loss_fcn = torch.nn.MSELoss(reduction='none')

    def __call__(self, pred, y):
        # prediction = self.downsample(pred)
        prediction = pred
        # target = self.downsample(y)
        target = y
        weights = 1 / (self.alpha - target) ** 2
        out = (prediction - target) ** 2
        out = out * weights
        return out.mean()
