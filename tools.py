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
    path = '/media/zhao/HD1/data/henan/'
    df = pd.read_csv(path + 'GE/GE_all.csv')
    error_item = pd.DataFrame(columns=['image_path'])
    for i in tqdm(range(len(df))):
        imagepath = path + df.loc[i, 'image_path']
        ### 窗宽窗位调整
        # try:
        dcm = pydicom.dcmread(imagepath)
        image = dcm.pixel_array
        # 通过窗宽窗位归一化
        # wc = int(dcm['0028', '1050'][2]) - 200
        # ww = int(dcm['0028', '1051'][2])
        # image = image.astype(np.int16)
        # image = 1 / (1 + np.exp(-4 * (image - wc) / ww))
        # image = (image * 255).astype(np.uint8)
        image = (image / 4095 * 65535).astype(np.uint16)

        cv2.imwrite(imagepath[0:-3] + 'png', image)
        # print(imagepath)


def crop_img():
    path = '/media/zhao/HD1/data/henan/'
    df = pd.read_csv(path + 'GE/GE_all_png.csv')
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
        imagepath = imagepath.replace('GE', 'GE_16')
        os.makedirs('/'.join(imagepath.split('/')[0:-1]), exist_ok=True)
        cv2.imwrite(imagepath, image)
        # cv2.imwrite('/home/zhao/mydata/projects/MSVCL_MICCAI2021/datasets/3/' + imagepath.split('/')[-1], image)
        # if i == 100:
        #     break


def compute_x():
    # path = '/home/zhao/mydata/datasets/GE/crop/'
    # df = pd.read_csv(path + 'GE_DEL_U.csv')
    path = '/home/zhao/mydata/datasets/HLG/crop/'
    df = pd.read_csv(path + 'HLG_del_U.csv')
    y1, y2, y3, y4 = 0, 0, 0, 0
    for i in tqdm(range(len(df))):
        imagepath = path + df.loc[i, 'image_path']
        image = cv2.imread(imagepath, -1) / 255
        x1 = image.sum() / (image != 0).sum()
        image2 = image * image
        x2 = image2.sum() / (image2 != 0).sum()
        image3 = image2 * image
        x3 = image3.sum() / (image3 != 0).sum()
        image4 = image3 * image
        x4 = image4.sum() / (image4 != 0).sum()
        y1 += x1
        y2 += x2
        y3 += x3
        y4 += x4
    y1 /= len(df)
    y2 /= len(df)
    y3 /= len(df)
    y4 /= len(df)
    print(y1, y2, y3, y4)


def copy_img():
    path = '/home/zhao/mydata/datasets/GE/crop/'
    df = pd.read_csv(path + 'GE_DEL_U.csv')
    for i in tqdm(range(1000)):
        imagepath = path + df.loc[i, 'image_path']
        image = cv2.imread(imagepath, -1)
        cv2.imwrite(
            '/home/zhao/mydata/projects/MSVCL_MICCAI2021/datasets/my_G2H/trainA/' +
            df.loc[i, 'image_path'].split(sep='/')[-1],
            image)
        # if i == 1000:
        #     break


def compute():
    path = '/home/zhao/mydata/datasets/cyclegan/transfer/trainA/'
    lists = os.listdir(path)
    y1, y2 = 0, 0
    for n in tqdm(lists):
        image = cv2.imread(path + n, -1)
        x1 = image.sum() / (image != 0).sum()
        image2 = image * image
        x2 = image2.sum() / (image2 != 0).sum()
        y1 += x1
        y2 += x2
    y1 /= 1000
    y2 /= 1000
    std = np.sqrt(y2 - y1 ** 2)
    print(y1, std)


def trans8():
    path = '/home/zhao/mydata/datasets/cyclegan/transfer512/HLG_full/'
    path1 = '/home/zhao/mydata/datasets/cyclegan/transfer512/B/'
    lists = os.listdir(path)
    for n in tqdm(lists):
        image = cv2.imread(path + n, -1)
        if image.shape[1] >= 512:
            cv2.imwrite(path1 + n, image)


if __name__ == '__main__':
    path = '/media/zhao/HD1/work_zhao/sig/save_img/GE/0.png'
    path1 = '/media/zhao/HD1/work_zhao/sig/save_img/GE/0_new.png'
    img = cv2.imread(path, -1)
    img1 = cv2.imread(path1, -1)
    a=1
