import os
import glob
import random
from config import cfg
import sys
from tqdm import tqdm
import numpy as np


def main(ratio=0.9):
    train_data_path = '../dataset/train'
    labels = os.listdir(train_data_path)
    test_data_path = '../dataset/test'

    train_img_list = []
    train_lab_list = []
    val_img_list = []
    val_lab_list = []

    # 读取image 和 label
    for index, label in tqdm(enumerate(labels)):
        img_list = glob.glob(os.path.join(train_data_path, label, '*.jpg'))
        lab_list = [index] * len(img_list)
        random.shuffle(img_list)
        train_img_list.extend(img_list[:int(ratio * len(img_list))])
        train_lab_list.extend(lab_list[:int(ratio * len(img_list))])
        val_img_list.extend(img_list[(int(ratio * len(img_list)) + 1):])
        val_lab_list.extend(lab_list[(int(ratio * len(img_list)) + 1):])

    train_data = np.vstack((train_img_list, train_lab_list))
    val_data = np.vstack((val_img_list, val_lab_list))
    train_data = train_data.transpose()
    val_data = val_data.transpose()
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    with open('.' + cfg.TRAIN_LABEL_DIR, 'w')as f:
        for index, img in enumerate(train_data[:, 0]):
            # print(img + ' ' + str(index))
            f.write(img[1:] + ' ' + str(int(train_data[index, 1:])))
            f.write('\n')

    with open('.' + cfg.VAL_LABEL_DIR, 'w')as f:
        for index, img in enumerate(val_data[:, 0]):
            # print(img + ' ' + str(index))
            f.write(img[1:] + ' ' + str(int(val_data[index, 1:])))
            f.write('\n')

    imglist = glob.glob(os.path.join(test_data_path, '*.jpg'))
    with open('.' + cfg.TEST_LABEL_DIR, 'w')as f:
        for img in imglist:
            f.write(img[1:])
            f.write('\n')


if __name__ == '__main__':
    main(ratio=0.9)