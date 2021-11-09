from abc import ABC

from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
import numpy as np
import torchvision.transforms as transforms


def get_transform(data):
    size = [1024, 512]
    if data == 'train':
        transform_list = [transforms.Resize(size),
                          # transforms.RandomHorizontalFlip(),
                          # transforms.RandomRotation(10)
                          ]
    else:
        transform_list = [transforms.Resize(size)]

    transform_list += [transforms.ToTensor()]
    # transform_list += [transforms.Normalize((0.5,), (0.5,))]

    return transforms.Compose(transform_list)


def make_pathology(pathology):
    if pathology == 'M':
        return 1
    elif pathology == 'MALIGNANT':
        return 1
    else:
        return 0


class GE_dataset(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, df, data):
        # df = pd.DataFrame()
        df.reset_index(drop=True, inplace=True)
        df.loc[:, 'image_path'] = pathImageDirectory + df['image_path']
        df['pathology_label'] = df['pathology_label'].apply(make_pathology)

        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.transform = get_transform(data)

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        imagePath = self.df.loc[index, 'image_path']
        imageLabel = [self.df.loc[index, 'pathology_label']]
        imageLabel = torch.FloatTensor(imageLabel)
        image = Image.open(imagePath).convert("I")
        image = self.transform(image)
        image = image / 65535.0
        # image = transforms.Normalize((0.5,), (0.5,))(image)
        return image, imageLabel

    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.df)
