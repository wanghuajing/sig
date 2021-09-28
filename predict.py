import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
import cv2
from model import *
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(640),
         transforms.ToTensor()
         ])

    # load image
    dir = '/home/zhao/mydata/datasets/GE/'
    pre_df = pd.read_csv(dir + 'GE.csv')
    save_df = pd.DataFrame(columns=['image_path', 'gt_wc', 'pr_wc', 'gt_ww', 'pr_ww'])
    for i, cow in tqdm(pre_df.iterrows()):
        img = pydicom.dcmread(dir + cow['image_path']).pixel_array
        img = (img / 4096).astype(np.float32)
        img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 256).astype(np.uint8)
        img = Image.fromarray(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # create model
        model = resnet50(num_classes=2)
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.to(device)
        # load model weights
        model_weight_path = "./resnet50_3_rgb/best.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = (torch.squeeze(model(img.to(device))).cpu().numpy() * 4096).astype(np.int16)
        data = {'image_path': cow['image_path'], 'gt_wc': cow['normal_wc'], 'pr_wc': output[0],
                'gt_ww': cow['normal_ww'], 'pr_ww': output[1]}
        save_df = save_df.append(data, ignore_index=True)
    save_df.to_csv('./csv/resnet50_3_csv_all.csv', index=False)


if __name__ == '__main__':
    main()
