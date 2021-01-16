import numpy as np
import random
import PIL.Image
import torch
import torchvision
from utils.data_aug.cutout import Cutout, DualCutout
from utils.data_aug.random_erasing import RandomErasing
from typing import Callable, Tuple, Union


class CenterCrop:
    def __init__(self, config):
        self.transform = torchvision.transforms.CenterCrop(
            config.DATASET.IMAGE_SIZE)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class RandomCrop:
    def __init__(self, config):
        self.transform = torchvision.transforms.RandomCrop(
            config.DATASET.IMAGE_SIZE,
            padding=config.AUGMENTATION.RANDOM_CROP.PADDING,
            fill=config.AUGMENTATION.RANDOM_CROP.FILL,
            padding_mode=config.AUGMENTATION.RANDOM_CROP.PADDING_MODE)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomResizeCrop:
    def __init__(self, config):
        self.transform = torchvision.transforms.RandomResizedCrop(
            config.DATASET.IMAGE_SIZE,
            scale=config.AUGMENTATION.RANDOM_RESIZE_CROP.SCALE,
            ratio=config.AUGMENTATION.RANDOM_RESIZE_CROP.RATIO)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self, config):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.AUGMENTATION.RANDOM_HORIZONTAL_FLIP.PROB)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Resize:
    def __init__(self, config):
        self.transform = torchvision.transforms.Resize(config.TTA.RESIZE)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class ColorJitter:
    def __init__(self, config):
        self.p = config.AUGMENTATION.COLOR_JITTER.PROB
        self.transform = torchvision.transforms.ColorJitter(brightness=config.AUGMENTATION.COLOR_JITTER.BRIGHTNESS,
                                                            contrast=config.AUGMENTATION.COLOR_JITTER.BRIGHTNESS,
                                                            saturation=config.AUGMENTATION.COLOR_JITTER.SATURATION,
                                                            hue=config.AUGMENTATION.COLOR_JITTER.HUE)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            data = self.transform(data)
        return data


class ToTensor:
    def __call__(
        self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))


class RandomRotate(object):
    def __init__(self, config):
        self.degree = config.AUGMENTATION.RANDOM_ROTATE.DEGREE
        self.p = config.AUGMENTATION.RANDOM_ROTATE.PROB

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, PIL.Image.BILINEAR)
        return img


class RandomGaussianBlur(object):
    def __init__(self, config):
        self.p = config.AUGMENTATION.USE_GASSIAN_BLUR.PROB

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(PIL.ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


def get_dataset_stats(config) -> Tuple[np.ndarray, np.ndarray]:
    name = config.DATASET.NAME
    if name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif name == 'CIFAR100':
        # RGB
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
    elif name == 'MNIST':
        mean = np.array([0.1307])
        std = np.array([0.3081])
    elif name == 'FashionMNIST':
        mean = np.array([0.2860])
        std = np.array([0.3530])
    elif name == 'KMNIST':
        mean = np.array([0.1904])
        std = np.array([0.3475])
    elif name == 'ImageNet':
        # RGB
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def create_transform(config, is_train: bool) -> Callable:
    mean, std = get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.AUGMENTATION.USE_COLOR_JITTER:
            transforms.append(ColorJitter(config))

        if config.AUGMENTATION.USE_RANDOM_RESIZE_CROP:
            transforms.append(RandomResizeCrop(config))
        # else:
        #     transforms.append(CenterCrop(config))
        if config.AUGMENTATION.USE_RANDOM_HORIZONTAL_FLIP:
            transforms.append(RandomHorizontalFlip(config))

        if config.AUGMENTATION.USE_RANDOM_ROTATE:
            transforms.append(RandomRotate(config))
        if config.AUGMENTATION.USE_GASSIAN_BLUR:
            transforms.append(RandomGaussianBlur(config))
        # if config.AUGMENTATION.USE_GRIDMASK:
        #     transforms.append(GridMask(config))

        transforms.append(Normalize(mean, std))

        if config.AUGMENTATION.USE_CUTOUT:
            transforms.append(Cutout(config))
        if config.AUGMENTATION.USE_RANDOM_ERASING:
            transforms.append(RandomErasing(config))
        if config.AUGMENTATION.USE_DUAL_CUTOUT:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = []
        if config.TTA.USE_RESIZE:
            transforms.append(Resize(config))
        if config.TTA.USE_CENTER_CROP:
            transforms.append(CenterCrop(config))
        transforms += [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)

