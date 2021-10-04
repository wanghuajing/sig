import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pydicom
from PIL import Image


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


def crop_img():
    path = '/home/zhao/mydata/datasets/GE/origin/'
    df = pd.read_csv(path + 'GE_all.csv')
    for i in tqdm(range(len(df))):
        imagepath = path + df.loc[i, 'image_path']
        image = cv2.imread(imagepath, -1)
        thre = 1
        kernel = np.ones((3, 3), np.uint8)
        # 阈值分割，pix大于thre=1的设置为255，低于thre设置为0
        while True:
            ret, mask = cv2.threshold(image, thre, 255, cv2.THRESH_BINARY)
            # 判断分割效果
            if np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) > 0.85:
                thre += 1
            else:
                break
        mask = mask.astype('uint8')
        mask = cv2.erode(mask, kernel, iterations=8)
        mask = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT, iterations=12)
        mask = cv2.erode(mask, kernel, iterations=30)
        # 挑选最大的连通区域
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_idx = 0
        for idx, contour in enumerate(contours):
            area = contour.shape[0]
            if area > max_area:  # 通过轮廓点的数目来找面积最大的肿块
                max_idx = idx
                max_area = area

        mask_max = np.zeros_like(mask)
        cv2.drawContours(mask_max, [contours[max_idx]], -1, 255, thickness=-1)
        # cv2.imwrite('mask_max.png', mask_max)
        # 膨胀操作
        mask = cv2.dilate(mask_max, kernel, borderType=cv2.BORDER_CONSTANT, iterations=90)
        locs = np.where(mask == 255)
        x0 = np.min(locs[1])
        x1 = np.max(locs[1])
        y0 = np.min(locs[0])
        y1 = np.max(locs[0])
        mask = mask / 255
        # image = image * mask
        image = image[y0:y1, x0:x1]
        imagepath = imagepath.replace('origin', 'crop')
        os.makedirs('/'.join(imagepath.split('/')[0:-1]), exist_ok=True)
        cv2.imwrite(imagepath, image)
        # cv2.imwrite('/home/zhao/mydata/datasets/GE/crop/'+imagepath.split('/')[-1],image)


if __name__ == '__main__':
    crop_img()
