import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def add_name():
    path = '/home/zhao/mydata/datasets/cbis_ddsm/cbis_ddsm_original_compressed/csv/val.csv'
    df = pd.read_csv(path)
    for i in tqdm(range(len(df))):
        df.loc[i, 'image_path'] = 'Mass-test/' + df.loc[i, 'image_path']
    df.to_csv('/home/zhao/mydata/datasets/cbis_ddsm/cbis_ddsm_original_compressed/csv/val1.csv')


if __name__ == '__main__':
    add_name()
