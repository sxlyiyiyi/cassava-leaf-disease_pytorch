import numpy as np


class Cutout:
    def __init__(self, config):
        aug_config = config.AUGMENTATION.CUTOUT
        self.p = aug_config.PROB
        self.mask_size = aug_config.MASK_SIZE
        self.cutout_inside = aug_config.CUT_INSIDE
        self.mask_color = aug_config.MASK_COLOR

        self.mask_size_half = aug_config.MASK_SIZE // 2
        self.offset = 1 if aug_config.MASK_SIZE % 2 == 0 else 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if self.cutout_inside:
            cxmin = self.mask_size_half
            cxmax = w + self.offset - self.mask_size_half
            cymin = self.mask_size_half
            cymax = h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image


class DualCutout:
    def __init__(self, config):
        self.cutout = Cutout(config)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.hstack([self.cutout(image), self.cutout(image)])
