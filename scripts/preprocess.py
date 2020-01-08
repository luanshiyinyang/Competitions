"""
Author: Zhou Chen
Date: 2020/1/8
Desc: 预处理数据集
"""
from config import data_folder, generate_folder
import os
import numpy as np
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import pandas as pd

import face_recognition


if not os.path.exists(generate_folder):
    os.mkdir(generate_folder)


def scale_img(img, scale_size=139):
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = scale_size * h / w, scale_size
    else:
        new_h, new_w = scale_size, scale_size * w / h

    new_h, new_w = int(new_h), int(new_w)
    img = cv2.resize(img, (new_w, new_h))
    top = None
    left = None
    if h == w:
        return img
    elif h < w:
        if new_w > scale_size:
            left = np.random.randint(0, new_w - scale_size)
        else:
            left = 0
        top = 0

    elif h > w:
        if new_h > scale_size:
            top = np.random.randint(0, new_h - scale_size)
        else:
            top = 0
        left = 0
    img = img[top: top + scale_size, left: left + scale_size]
    return img


def preprocess():
    categories = os.listdir(data_folder)
    for category in categories:
        in_path = os.path.join(data_folder, category)
        out_path = os.path.join(generate_folder, category + '_face')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for file in glob(in_path + '/*.jpg'):
            file_name = file.split('\\')[-1]
            print(file_name)
            img = face_recognition.load_image_file(file)
            if max(img.shape) > 2000:
                if img.shape[0] > img.shape[1]:
                    img = cv2.resize(img, (2000, int(2000 * img.shape[1] / img.shape[0])))
                else:
                    img = cv2.resize(img, (int(2000 * img.shape[0] / img.shape[1]), 2000))
            locations = face_recognition.face_locations(img)  # 人脸检测，大部分为单个，但也有多个检测结果
            if len(locations) <= 0:
                print("no face")
            else:
                for i, (a, b, c, d) in enumerate(locations):
                    image_split = img[a:c, d:b, :]
                    image_split = scale_img(image_split)
                    Image.fromarray(image_split).save(os.path.join(out_path, file_name + '_{}.png'.format(i)))


def generate_desc_csv(folder):
    categories = os.listdir(folder)
    file_id = []
    label = []
    for category in tqdm(categories):
        images = glob(os.path.join(folder, category) + '/*.jpg')  # 这里可以观察一下数据集，发现图片均为jpg格式，正则匹配比较简单
        for img in images:
            file_id.append(img)
            label.append(category)
    df_description = pd.DataFrame({'file_id': file_id, 'label': label})
    df_description.to_csv('../data/description.csv', encoding='utf8', index=False)  # 落地这个csv文件是为了更符合常见的数据集说明文件


if __name__ == '__main__':
    # preprocess()  # 生成处理后的数据集
    generate_desc_csv(generate_folder)  # 为生成的数据集生成说明文件，方便后面keras的生成器读取