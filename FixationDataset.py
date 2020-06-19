from __future__ import print_function
from torch.utils.data import Dataset
import os
import imageio
import torch
import numpy as np


class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, transform=None, testing=False):
        self.root_dir = root_dir
        self.image_files = self.read_file(image_file)
        self.fixation_files = self.read_file(fixation_file)
        self.transform = transform
        self.testing = testing
        if not self.testing:
            assert (len(self.image_files) == len(self.fixation_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        if len(image.shape) == 2:
            print(img_name)

        if self.testing:
            fix = np.zeros((224, 224))
        else:
            fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
            fix = imageio.imread(fix_name)

        if self.testing:
            sample = {'image': image, 'fixation': fix, 'name': self.image_files[idx].split("/")[2]}
        else:
            sample = {'image': image, 'fixation': fix}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def read_file(filename):
        lines = []
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()  # or some other preprocessing
                lines.append(line)
        return lines