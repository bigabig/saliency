import numpy as np
import torch
import torchvision


class Rescale:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        image = image.astype(np.float32) / 255.0
        fixation = fixation.astype(np.float32) / 255.0
        return {'image': image, 'fixation': fixation}


class ToTensor:
    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']

        # reshape image
        height, width, channels = image.shape
        image = image.reshape((channels, height, width))

        # reshape fixation
        fixation = np.expand_dims(fixation, 0)

        # convert to torch tensors
        image = torch.from_numpy(image)
        fixation = torch.from_numpy(fixation)

        return {'image': image, 'fixation': fixation}


class Normalize:
    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return {'image': image, 'fixation': fixation}
