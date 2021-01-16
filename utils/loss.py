import torch
import torch.nn as nn
from typing import Callable, Tuple, List
import torch.nn.functional as F


def create_loss(config) -> Tuple[Callable, Callable]:
    if config.AUGMENTATION.USE_MIXUP:
        train_loss = MixupLoss(reduction='mean')
    elif config.AUGMENTATION.USE_RICAP:
        train_loss = RICAPLoss(reduction='mean')
    elif config.AUGMENTATION.USE_CUTMIX:
        train_loss = CutMixLoss(reduction='mean')
    elif config.AUGMENTATION.USE_LABEL_SMOOTHING:
        train_loss = LabelSmoothingLoss(config, reduction='mean')
    elif config.AUGMENTATION.USE_DUAL_CUTOUT:
        train_loss = DualCutoutLoss(config, reduction='mean')
    else:
        train_loss = nn.CrossEntropyLoss(reduction='mean')
    val_loss = nn.CrossEntropyLoss(reduction='mean')
    return train_loss, val_loss


class MixupLoss:
    def __init__(self, reduction: str):
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        targets1, targets2, lam = targets
        return lam * self.loss_func(predictions, targets1) + (
            1 - lam) * self.loss_func(predictions, targets2)


def onehot_encoding(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)


def cross_entropy_loss(data: torch.Tensor, target: torch.Tensor,
                       reduction: str) -> torch.Tensor:
    logp = F.log_softmax(data, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


class LabelSmoothingLoss:
    def __init__(self, config, reduction: str):
        self.n_classes = config.DATASET.N_CLASSES
        self.epsilon = config.AUGMENTATION.LABEL_SMOOTHING.EPSILON
        self.reduction = reduction

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        device = predictions.device

        onehot = onehot_encoding(
            targets, self.n_classes).type_as(predictions).to(device)
        targets = onehot * (1 - self.epsilon) + torch.ones_like(onehot).to(
            device) * self.epsilon / self.n_classes
        loss = cross_entropy_loss(predictions, targets, self.reduction)
        return loss


class RICAPLoss:
    def __init__(self, reduction: str):
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[List[torch.Tensor], List[float]]) -> torch.Tensor:
        target_list, weights = targets
        return sum([
            weight * self.loss_func(predictions, targets)
            for targets, weight in zip(target_list, weights)
        ])


class CutMixLoss:
    def __init__(self, reduction: str):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        targets1, targets2, lam = targets
        return lam * self.criterion(predictions, targets1) + (
            1 - lam) * self.criterion(predictions, targets2)


class DualCutoutLoss:
    def __init__(self, config, reduction: str):
        self.alpha = config.augmentation.cutout.dual_cutout_alpha
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        predictions1, predictions2 = predictions[:, 0], predictions[:, 1]
        return (self.loss_func(predictions1, targets) + self.loss_func(
            predictions2, targets)) * 0.5 + self.alpha * F.mse_loss(
                predictions1, predictions2)
