import torch
import numpy as np
from utils.data_aug.transforms import get_dataset_stats
import matplotlib.pyplot as plt


def show_image(images, labels, config):
    mean, std = get_dataset_stats(config)

    # images = images.cpu().float().numpy()
    images = images.float().numpy()
    for i in range(images.shape[0]):
        img = images[i]
        for j in range(len(mean)):
            img[j] = img[j] * std[j] + mean[j]  # unnormalize
        img = (img * 255).astype(int)
        plt.figure()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()



