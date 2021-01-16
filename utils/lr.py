import torch

from utils.scheduler.combined_scheduler import CombinedScheduler
from utils.scheduler.components import (ConstantScheduler,
                                        CosineScheduler,
                                        ExponentialScheduler,
                                        LinearScheduler)
from utils.scheduler.multistep_scheduler import MultistepScheduler
from utils.scheduler.sgdr import SGDRScheduler


def _create_warmup(config, warmup_steps):
    warmup_type = config.SCHEDULER.WARMUP.TYPE
    if warmup_type == 'none' or warmup_steps == 0:
        return None

    warmup_start_factor = config.SCHEDULER.WARMUP.START_FACTOR

    if warmup_type == 'linear':
        lr_end = 1
        lr_start = warmup_start_factor
        scheduler = LinearScheduler(warmup_steps, lr_start, lr_end)
    elif warmup_type == 'exponential':
        scheduler = ExponentialScheduler(warmup_steps, config.TRAIN.BASE_LR,
                                         config.SCHEDULER.WARMUP.EXPONENT,
                                         warmup_start_factor)
    else:
        raise ValueError()

    return scheduler


def _create_main_scheduler(config, main_steps):
    scheduler_type = config.SCHEDULER.TYPE

    if scheduler_type == 'constant':
        scheduler = ConstantScheduler(main_steps, 1)
    elif scheduler_type == 'multistep':
        lr_decay = config.SCHEDULER.LR_DECAY
        scheduler = MultistepScheduler(main_steps, 1, lr_decay,
                                       config.SCHEDULER.MILESTONES)
    elif scheduler_type == 'linear':
        lr_start = 1
        lr_end = config.SCHEDULER.LR_MIN_FACTOR
        scheduler = LinearScheduler(main_steps, lr_start, lr_end)
    elif scheduler_type == 'cosine':
        scheduler = CosineScheduler(main_steps, 1,
                                    config.SCHEDULER.LR_MIN_FACTOR)
    elif scheduler_type == 'sgdr':
        scheduler = SGDRScheduler(main_steps, 1, config.SCHEDULER.T0,
                                  config.SCHEDULER.T_MUL,
                                  config.SCHEDULER.LR_MIN_FACTOR)
    else:
        raise ValueError()

    return scheduler


def create_scheduler(config, optimizer, steps_per_epoch):
    warmup_epochs = config.SCHEDULER.WARMUP.EPOCHS
    main_epochs = config.SCHEDULER.EPOCHS - warmup_epochs

    warmup_scheduler = _create_warmup(config, warmup_epochs)
    main_scheduler = _create_main_scheduler(config, main_epochs)

    scheduler_func = CombinedScheduler([warmup_scheduler, main_scheduler])
    scheduler_func.multiply_steps(steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func, config.TRAIN.RESUME_EPOCH + 1)

    return scheduler
