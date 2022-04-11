import os
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, index):
        x = cv2.imread(self.images_path[index])
        x = preprocess_image(x)
        
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.images_path)

def build_dataset(args):
    images_path = [args.folder + '/' + im for im in os.listdir(args.folder) if im.endswith('.png')]
    transform = transforms.ToTensor()

    return CarlaDataset(images_path, transform), None

def preprocess_image(image, convert_to_rgb=False):
    """
    Crop, resize and normalize image.
    Optionnally it also converts the image from BGR to RGB.
    :param image: (np.ndarray) image (BGR or RGB)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :return: (np.ndarray)
    """
    # Crop
    # Region of interest
    image = image[400:, :]
    # Resize
    im = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im
