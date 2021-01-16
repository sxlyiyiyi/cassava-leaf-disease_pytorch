import torch
from utils.optim.adabound import AdaBoundW, AdaBound
from utils.optim.lars import LARSOptimizer
from typing import Tuple
import thop
from easydict import EasyDict as edict
import random
import numpy as np


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model


def create_optimizer(config, model):
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=config.TRAIN.BASE_LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    nesterov=config.TRAIN.NESTEROV)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()),
                                       'initial_lr': config.TRAIN.BASE_LR}],
                                     lr=config.TRAIN.BASE_LR,
                                     betas=config.OPTIM.ADAM.BETAS)
    elif config.TRAIN.OPTIMIZER == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config.TRAIN.BASE_LR,
                                     betas=config.OPTIM.ADAM.BETAS,
                                     amsgrad=True)
    elif config.TRAIN.OPTIMIZER == 'adabound':
        optimizer = AdaBound(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config.TRAIN.BASE_LR,
                             betas=config.OPTIM.ADABOUND.BETAS,
                             final_lr=config.OPTIM.ADABOUND.FINAL_LR,
                             gamma=config.OPTIM.ADABOUND.GAMMA)
    elif config.TRAIN.OPTIMIZER == 'adaboundw':
        optimizer = AdaBoundW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config.TRAIN.BASE_LR,
                              betas=config.OPTIM.ADABOUND.BETAS,
                              final_lr=config.OPTIM.ADABOUND.FINAL_LR,
                              gamma=config.OPTIM.ADABOUND.GAMMA)
    elif config.TRAIN.OPTIMIZER == 'lars':
        optimizer = LARSOptimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config.TRAIN.BASE_LR,
                                  momentum=config.TRAIN.MOMENTUM,
                                  eps=config.OPTIM.LARS.EPS,
                                  thresh=config.OPTIM.LARS.THRESHOLD)
    else:
        raise ValueError()
    return optimizer


def subdivide_batch(config, data, targets):
    subdivision = config.TRAIN.SUBDIVISION

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.AUGMENTATION.USE_MIXUP or config.AUGMENTATION.USE_CUTMIX:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in zip(
            targets1.chunk(subdivision), targets2.chunk(subdivision))]
    elif config.AUGMENTATION.USE_RICAP:
        target_list, weights = targets
        target_list_chunks = list(
            zip(*[target.chunk(subdivision) for target in target_list]))
        target_chunks = [(chunk, weights) for chunk in target_list_chunks]
    else:
        target_chunks = targets.chunk(subdivision)
    return data_chunks, target_chunks


def send_targets_to_device(config, targets, device):
    if config.AUGMENTATION.USE_MIXUP or config.AUGMENTATION.USE_CUTMIX:
        t1, t2, lam = targets
        targets = (t1.to(device), t2.to(device), lam)
    elif config.AUGMENTATION.USE_RICAP:
        labels, weights = targets
        labels = [label.to(device) for label in labels]
        targets = (labels, weights)
    else:
        targets = targets.to(device)
    return targets


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def count_op(config, model: torch.nn.Module) -> Tuple[str, str]:
    data = torch.zeros((1, config.DATASET.N_CHANNELS,
                        config.DATASET.IMAGE_SIZE[0], config.DATASET.IMAGE_SIZE[1]),
                       dtype=torch.float32,
                       device=torch.device(config.DEVICE))
    return thop.clever_format(thop.profile(model, (data, ), verbose=False))


def get_env_info(config):
    info = {
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda or '',
        'cudnn_version': torch.backends.cudnn.version() or '',
    }
    if config.DEVICE != 'cpu':
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info['gpu_capability'] = f'{capability[0]}.{capability[1]}'

    return edict({'env_info': info})


def set_seed(config) -> None:
    seed = config.TRAIN.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_cudnn(config) -> None:
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC

