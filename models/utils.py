try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# home = os.path.expanduser('~')
home = '.'
# 预训练模型的存放位置
LOCAL_PRETRAINED = {
    'resnext101_32x8d': home + '/weights/resnext101_32x8.pth',
    # 'resnext101_32x16d': home + '/weights/resnext101_32x16.pth',
    'resnext101_32x48d': home + '/weights/resnext101_32x48.pth',
    # 'resnext101_32x32d': home + '/weights/resnext101_32x32.pth',
    'resnext101_32x32d': None,
    'resnet50': home + '/weights/resnet50.pth',
    'resnet101': home + '/weights/resnet101.pth',
    'densenet121': home + '/weights/densenet121.pth',
    'densenet169': home + '/weights/densenet169.pth',
    'moblienetv2': home + '/weights/mobilenetv2.pth',
    'efficientnet-b0': home + '/weights/efficientnet-b0.pth',
    'efficientnet-b1': home + '/weights/efficientnet-b1.pth',
    'efficientnet-b2': home + '/weights/efficientnet-b2.pth',
    'efficientnet-b3': home + '/weights/efficientnet-b3.pth',
    'efficientnet-b4': home + '/weights/efficientnet-b4.pth',
    'efficientnet-b5': home + '/weights/efficientnet-b5.pth',
    'efficientnet-b6': home + '/weights/efficientnet-b6.pth',
    'efficientnet-l2': home + '/weights/efficientnet-l2.pth',
    'efficientnet-b7': home + '/weights/efficientnet-b7.pth',
    'efficientnet-b8': home + '/weights/efficientnet-b8.pth'

}

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x8d_wsl': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d_wsl': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x48d_wsl': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x32d_wsl': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'moblienet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenet_v2_x1_5': None,
    'shufflenet_v2_x2_0': None,
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
    'efficientnet-b8': None,
    'efficientnet-l2': None
}
