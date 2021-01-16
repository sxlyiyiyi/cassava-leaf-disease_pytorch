import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Callable, Tuple, Union
import os
from PIL import Image
from utils.data_aug.transforms import create_transform
from utils.data_aug.cutmix import CutMixCollator
from utils.data_aug.mixup import MixupCollator
from utils.data_aug.ricap import RICAPCollator
import sys


def create_collator(config) -> Callable:
    if config.AUGMENTATION.USE_MIXUP:
        return MixupCollator(config)
    elif config.AUGMENTATION.USE_RICAP:
        return RICAPCollator(config)
    elif config.AUGMENTATION.USE_CUTMIX:
        return CutMixCollator(config)
    else:
        return torch.utils.data.dataloader.default_collate


class SelfCustomDataset(Dataset):
    def __init__(self, config, label_file, is_train):
        # 所有图片的绝对路径
        with open(label_file, 'r', encoding='utf-8') as f:
            # label_file的格式， （label_file image_label)
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        # 相关预处理的初始化
        #   self.transforms=transform
        self.img_aug = True
        self.transform = create_transform(config=config, is_train=is_train)
        self.input_size = config.DATASET.IMAGE_SIZE

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        label = np.array(label).astype(np.int64)
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.input_size)

        if self.img_aug:
            img = self.transform(img)

        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(label)

    def __len__(self):
        return len(self.imgs)


def create_dataloader(config, label_dir, batch_size, is_train):
    dataset = SelfCustomDataset(config, label_dir, is_train=is_train)
    if is_train:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2,
                                                 collate_fn=create_collator(config))
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader


if __name__ == "__main__":
    pass
