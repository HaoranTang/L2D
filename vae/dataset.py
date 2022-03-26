import os
import numpy as np
import torch

class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.images.shape[0]

def build_dataset(args):
    images = [im for im in os.listdir(args.folder) if im.endswith('.png')]
    images = torch.tensor(np.array(images))

    return CarlaDataset(images)

