import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pydicom


def add_name():
    path = '/home/zhao/mydata/datasets/cbis_ddsm/cbis_ddsm_original_compressed/csv/val.csv'
    df = pd.read_csv(path)
    for i in tqdm(range(len(df))):
        df.loc[i, 'image_path'] = 'Mass-test/' + df.loc[i, 'image_path']
    df.to_csv('/home/zhao/mydata/datasets/cbis_ddsm/cbis_ddsm_original_compressed/csv/val1.csv')


def dcm2png():
    path = '/home/zhao/mydata/datasets/GE/'
    df = pd.read_csv(path + 'GE_all.csv')
    error_item = pd.DataFrame(columns=['image_path'])
    for i in tqdm(range(len(df))):
        imagepath = path + df.loc[i, 'image_path']
        ### 窗宽窗位调整
        try:
            dcm = pydicom.dcmread(imagepath)
            image = dcm.pixel_array
            wc = int(dcm['0028', '1050'][0])
            ww = int(dcm['0028', '1051'][0])
            image = image.astype(np.int16)
            image = 1 / (1 + np.exp(-4 * (image - wc) / ww))
            image = (image * 255).astype(np.uint8)
            cv2.imwrite(imagepath[0:-3] + 'png', image)
        except:
            print(imagepath)
            error_item = error_item.append({'image_path': imagepath}, ignore_index=True)

    error_item.to_csv('error.csv', index=False)


if __name__ == '__main__':
    pass
