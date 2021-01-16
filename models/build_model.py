import os
import ast
import torch.nn as nn
import torch
import torchvision
from torch.hub import load_state_dict_from_url
from models.resnet import resnet101, resnet50, resnet18, resnet34, resnet152, resnext50_32x4d, resnext101_32x8d
from models.alexnet import alexnet
from models.densenet import densenet121, densenet169, densenet161, densenet201
from models.googlenet import googlenet
from models.inception import inception_v3
from models.mobilenet import mobilenet_v2
from models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from models.squeezenet import squeezenet1_0, squeezenet1_1
from models.vgg import vgg11, vgg16, vgg19, vgg13, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.efficientnet.model import EfficientNet
from models.resnext_wsl import resnext101_32x8d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl, resnext101_32x16d_wsl
from models.utils import LOCAL_PRETRAINED, model_urls
from pathlib import Path
from utils.utils import load_checkpoint


def Basenet(model_name, num_classes, test=False):
    model = eval(model_name)()
    if not test:
        if Path(LOCAL_PRETRAINED[model_name]).exists():
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        else:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)

        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


def Densenet(model_name, num_classes, test=False):
    model = eval(model_name)()
    if not test:
        if LOCAL_PRETRAINED[model_name] is None:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
            # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
            # 直接利用if语句筛选不一致的键
            #  修正键值的不对应
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Efficientnet(model_name, num_classes, test=False):
    """
    model_name :'efficientnet-b0', 'efficientnet-b1-7'
    """
    model = EfficientNet.from_name(model_name)
    if not test:
        if Path(LOCAL_PRETRAINED[model_name]).exists():
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        else:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model.load_state_dict(state_dict)
    fc_features = model._fc.in_features
    model._fc = nn.Linear(fc_features, num_classes)
    return model


def creat_model(config) -> nn.Module:
    if not config.TRAIN.RESUME_EPOCH:
        model = config.MODEL_NAMES[config.MODEL_NAME](config.MODEL_NAME, num_classes=config.DATASET.N_CLASSES)
    else:
        model = load_checkpoint(os.path.join(config.TRAIN.OUTPUT_DIR, 'epoch_{}.pth'.format(config.TRAIN.RESUME_EPOCH)))

    device = torch.device(config.DEVICE)
    model.to(device)

    return model


