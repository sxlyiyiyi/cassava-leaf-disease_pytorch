import numpy as np
from sklearn.metrics import f1_score, average_precision_score
import torch


class ClassMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.y_true = []
        self.y_pred = []

    def f1score(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

        return f1

    def acc(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        acc = np.sum(y_pred == y_true) / len(y_true)

        return acc

    def mAP(self):
        y_true = np.array(np.one_hot(self.y_true, self.num_class))
        y_pred = np.array(np.one_hot(self.y_pred, self.num_class))
        mAP = average_precision_score(y_true, y_pred)

        return mAP

    def addBatch(self, label, predict):
        assert predict.shape == label.shape
        self.y_true.extend(label)
        self.y_pred.extend(predict)

    def reset(self):
        self.y_true = []
        self.y_pred = []


def compute_accuracy(config, outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        if config.AUGMENTATION.USE_MIXUP or config.AUGMENTATION.USE_CUTMIX:
            targets1, targets2, lam = targets
            accs1 = accuracy(outputs, targets1, topk)
            accs2 = accuracy(outputs, targets2, topk)
            accs = tuple([
                lam * acc1 + (1 - lam) * acc2
                for acc1, acc2 in zip(accs1, accs2)
            ])
        elif config.AUGMENTATION.USE_RICAP:
            weights = []
            accs_all = []
            for labels, weight in zip(*targets):
                weights.append(weight)
                accs_all.append(accuracy(outputs, labels, topk))
            accs = []
            for i in range(len(accs_all[0])):
                acc = 0
                for weight, accs_list in zip(weights, accs_all):
                    acc += weight * accs_list[i]
                accs.append(acc)
            accs = tuple(accs)
        elif config.AUGMENTATION.USE_DUAL_CUTOUT:
            outputs1, outputs2 = outputs[:, 0], outputs[:, 1]
            accs = accuracy((outputs1 + outputs2) / 2, targets, topk)
        else:
            accs = accuracy(outputs, targets, topk)
    else:
        accs = accuracy(outputs, targets, topk)
    return accs


def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        batch_size = targets.size(0)

        pred = torch.max(outputs, 1)[1].t()
        correct = pred.eq(targets)

        res = correct.view(-1).float().sum(0, keepdim=True).mul_(1 / batch_size)
    return res


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num

    def acc(self):
        self.avg = self.sum / self.count
        return self.avg


