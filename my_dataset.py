from abc import ABC

from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
import numpy as np


def make_pathology(pathology):
    if pathology == 'M':
        return 1
    elif pathology == 'MALIGNANT':
        return 1
    else:
        return 0


class MyDataSet(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, df, transform):
        # df = pd.DataFrame()
        df.reset_index(drop=True, inplace=True)
        df.loc[:, 'image_path'] = pathImageDirectory + df['image_path']
        df['pathology_label'] = df['pathology_label'].apply(make_pathology)

        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        imagePath = self.df.loc[index, 'image_path']
        imageLabel = [self.df.loc[index, 'pathology_label']]

        imageData = cv2.imread(imagePath, -1)
        imageLabel = torch.LongTensor(imageLabel)
        imageData = Image.fromarray(imageData, 'F')
        imageInfo = self.df.loc[index].to_dict()

        if self.transform != None: imageData = self.transform(imageData)  # cp val test不使用

        return imageData, imageLabel, imageInfo

    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.df)


class cbis_ddsm(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, df, transform):
        # df = pd.DataFrame()
        df.reset_index(drop=True, inplace=True)
        df.loc[:, 'image_path'] = pathImageDirectory + df['image_path']
        df['pathology_label'] = df['pathology_label'].apply(make_pathology)

        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        imagePath = self.df.loc[index, 'image_path']
        imageLabel = [self.df.loc[index, 'pathology_label']]
        imageData = pydicom.dcmread(imagePath).pixel_array
        imageData = (imageData / (2 ** 16 - 1)).astype(np.float32)
        imageLabel = torch.LongTensor(imageLabel)
        imageData = Image.fromarray(imageData, 'F')

        if self.transform is not None: imageData = self.transform(imageData)  # cp val test不使用

        return imageData, imageLabel

    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.df)


class GE_dataset(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, df, transform):
        # df = pd.DataFrame()
        df.reset_index(drop=True, inplace=True)
        df.loc[:, 'image_path'] = pathImageDirectory + df['image_path']
        df['pathology_label'] = df['pathology_label'].apply(make_pathology)

        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        imagePath = self.df.loc[index, 'image_path']
        imageLabel = [self.df.loc[index, 'pathology_label']]
        # imageData = pydicom.dcmread(imagePath).pixel_array
        image = Image.open(imagePath)
        imageLabel = torch.LongTensor(imageLabel)
        # imageData = imageData / (2 ** 12 - 1)
        # imageData = Image.fromarray(imageData)
        image = image.resize((1280, 1280))
        image = np.asarray(image).astype(np.int16)
        image = np.dstack((image, image, image))
        image = image.transpose([2, 0, 1])
        image = (image / 4095).astype(np.float32)
        image = torch.tensor(image)
        # if self.transform is not None: imageData = self.transform(imageData)  # cp val test不使用

        return image, imageLabel

    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.df)
