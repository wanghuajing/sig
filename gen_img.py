import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
from model import *
from tqdm import tqdm
from models.sig_module import *
import cv2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = [1024, 512]
    data_transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor()
         ])
    save_path = './save_img/GE_loss/'
    os.makedirs('./save_img/GE_loss/',exist_ok=True)
    # load image
    dir = '/media/zhao/HD1/data/crop_GEHLG/'
    pre_df = pd.read_csv(dir + 'GE/GE_val.csv')

    for i, cow in tqdm(pre_df.iterrows()):
        img = Image.open(dir + cow['image_path'])
        img.save(save_path + '{}.png'.format(i))
        img = data_transform(img)
        img = img / 65535.0
        img = img[None, :, :, :]

        # create model
        model = get_image()
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        # load model weights
        model_weight_path = "./runs/GE_sig/GE_loss/e_50.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = model(img)
            img = (output[0, 0, :, :].cpu().numpy() * 65535.0).astype(np.uint16)
            cv2.imwrite(save_path + '{}_new.png'.format(i), img)


if __name__ == '__main__':
    main()
