import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from config import cfg

from utils.dataset import create_dataloader
from utils.utils import create_optimizer, load_checkpoint, send_targets_to_device, subdivide_batch, count_op
from utils.utils import get_env_info, setup_cudnn, set_seed
from utils.loss import create_loss
from utils.lr import create_scheduler
from utils.visualization import show_image
from tqdm import tqdm
from utils.logger import create_logger
from models.build_model import creat_model
from utils.metrics import ClassMetric, AverageMeter, compute_accuracy
from utils.tensorboard import create_tensorboard_writer


def main():

    set_seed(cfg)
    setup_cudnn(cfg)

    model = creat_model(cfg)

    output_dir = pathlib.Path(cfg.TRAIN.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')

    macs, n_params = count_op(cfg, model)
    logger.info(f'MACs   : {macs}')
    logger.info(f'#params: {n_params}')
    # logger.info(get_env_info(cfg))

    # 定义优化器与损失函数
    device = torch.device(cfg.DEVICE)
    optimizer = create_optimizer(cfg, model)
    train_loss_fun, val_loss_fun = create_loss(cfg)
    train_metrics = ClassMetric(cfg.DATASET.N_CLASSES)
    val_metrics = ClassMetric(cfg.DATASET.N_CLASSES)

    # region 数据集配置
    train_dataloader = create_dataloader(cfg, cfg.TRAIN_LABEL_DIR, cfg.TRAIN.BATCH_SIZE, is_train=True)
    val_dataloader = create_dataloader(cfg, cfg.VAL_LABEL_DIR, cfg.TRAIN.BATCH_SIZE, is_train=False)

    scheduler = create_scheduler(cfg,
                                 optimizer,
                                 steps_per_epoch=len(train_dataloader))
    # cosine学习率调整
    train_acc_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    for epoch in range(cfg.TRAIN.RESUME_EPOCH, cfg.TRAIN.MAX_EPOCH):
        model.train()

        for batch, (images, labels) in tqdm(enumerate(train_dataloader)):
            # show_image(images, labels, cfg)
            images = images.to(device)
            labels = labels.to(device)
            targets = send_targets_to_device(cfg, labels, device)
            data_chunks, target_chunks = subdivide_batch(cfg, images, targets)

            optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
            outputs = []
            losses = []
            for data_chunk, target_chunk in zip(data_chunks, target_chunks):
                if cfg.AUGMENTATION.USE_DUAL_CUTOUT:
                    w = data_chunk.size(3) // 2
                    data1 = data_chunk[:, :, :, :w]
                    data2 = data_chunk[:, :, :, w:]
                    outputs1 = model(data1)
                    outputs2 = model(data2)
                    output_chunk = torch.cat(
                        (outputs1.unsqueeze(1), outputs2.unsqueeze(1)), dim=1)
                else:
                    output_chunk = model(data_chunk)
                outputs.append(output_chunk)
                # out = model(data_chunks)
                loss = train_loss_fun(output_chunk, target_chunk)
                losses.append(loss)
                # loss = train_loss_fun(out, labels.clone().detach().long())
                loss.backward()  # loss反向传播
            outputs = torch.cat(outputs)

            if cfg.TRAIN.SUBDIVISION > 1:
                for param in model.parameters():
                    param.grad.data.div_(cfg.TRAIN.SUBDIVISION)

            optimizer.step()  # 梯度更新

            if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0 and epoch > 0:
                acc = compute_accuracy(cfg, outputs, targets, augmentation=True)
                loss = sum(losses)

                num = images.size(0)

                train_loss_meter.update(loss.item(), num)
                train_acc_meter.update(acc.item(), num)

        if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0 and epoch > 0:
            with torch.no_grad():
                for batch, (images, labels) in tqdm(enumerate(val_dataloader)):
                    model.eval()
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    out = model(images)
                    prediction = torch.max(out, 1)[1]
                    val_metrics.addBatch(labels.cpu().numpy(), prediction.cpu().numpy())

            logger.info("Epoch: {}, lr:{:.5f} train_f1:{:.5f}, val_f1{:.5f}, train_acc: {:.5f}, val_acc:{:.5f}".format(
                epoch, scheduler.get_last_lr()[0], train_acc_meter.acc(), val_metrics.f1score(), train_metrics.acc(), val_metrics.acc()))
        train_metrics.reset()
        val_metrics.reset()
        scheduler.step()

        # region 保存模型
        if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0 and epoch > 0:
            checkpoint = {'model': model,
                          'model_state_dict': model.state_dict(),
                          # 'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, os.path.join(output_dir, 'epoch_{}.pth'.format(epoch)))
        # endregion

        # if iteration % 10 == 0:
        #     print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
        #           + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' % (
        #                       train_acc * 100) + 'LR: %.8f' % (lr))


if __name__ == '__main__':
    main()