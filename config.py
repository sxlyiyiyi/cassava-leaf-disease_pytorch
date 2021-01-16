import os
from easydict import EasyDict as edict
from models.build_model import Basenet, Densenet, Efficientnet

__C = edict()
__C.TRAIN = edict()
cfg = __C

__C.MODEL_NAME = 'efficientnet-b4'                                        # 采用的模型名称
__C.RATIO = 0.9                                                    # 训练集验证集划分比例
__C.TRAIN_LABEL_DIR = './dataset/train.txt'                        # 数据集的存放位置
__C.VAL_LABEL_DIR = './dataset/val.txt'
__C.TEST_LABEL_DIR = './dataset/test.txt'
__C.DEVICE = 'cuda'
# cuDNN
__C.CUDNN = edict()
__C.CUDNN.BENCHMARK = True
__C.CUDNN.DETERMINISTIC = False

__C.TRAIN.OUTPUT_DIR = './checkpoint/' + __C.MODEL_NAME
__C.TRAIN.BATCH_SIZE = 48                                         # batch_size
__C.TRAIN.MAX_EPOCH = 500                                          # 最大迭代次数
__C.TRAIN.RESUME_EPOCH = 40                                         # 从第几个epoch开始resume训练，如果为0，从头开始
__C.TRAIN.SNAPSHOT_EPOCH = 5                                       # 保存间隔

__C.TRAIN.OPTIMIZER = 'adam'                                       # options: sgd, adam, lars, adabound, adaboundw
__C.TRAIN.BASE_LR = 0.0005                                           # 学习率
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.NESTEROV = True
__C.TRAIN.SUBDIVISION = 1
__C.TRAIN.SEED = 0

__C.DATASET = edict()
__C.DATASET.IMAGE_SIZE = (256, 256)                                      # 图片尺寸
__C.DATASET.DATASET_DIR = ''
__C.DATASET.N_CHANNELS = 3                                         # 输入图片通道数
__C.DATASET.N_CLASSES = 5                                         # 类别数
__C.DATASET.LABLES = ['calling', 'normal', 'smoking', 'smoking_calling']
__C.DATASET.NAME = 'ImageNet'

# 预处理
__C.AUGMENTATION = edict()
__C.AUGMENTATION.USE_COLOR_JITTER = True
__C.AUGMENTATION.USE_RANDOM_ROTATE = False                                  # mixup
__C.AUGMENTATION.USE_MIXUP = False                                  # mixup
__C.AUGMENTATION.USE_RICAP = False                                  # 随机图像裁剪和修补
__C.AUGMENTATION.USE_GASSIAN_BLUR = False
__C.AUGMENTATION.USE_CUTMIX = False
__C.AUGMENTATION.USE_RANDOM_RESIZE_CROP = False
__C.AUGMENTATION.USE_LABEL_SMOOTHING = False
__C.AUGMENTATION.USE_DUAL_CUTOUT = False
__C.AUGMENTATION.USE_RANDOM_HORIZONTAL_FLIP = False
__C.AUGMENTATION.USE_CUTOUT = False
__C.AUGMENTATION.USE_RANDOM_ERASING = False
__C.AUGMENTATION.USE_GRIDMASK = False

__C.AUGMENTATION.COLOR_JITTER = edict()
__C.AUGMENTATION.COLOR_JITTER.PROB = 0.5
__C.AUGMENTATION.COLOR_JITTER.BRIGHTNESS = 0.3
__C.AUGMENTATION.COLOR_JITTER.CONTRAST = 0.3
__C.AUGMENTATION.COLOR_JITTER.SATURATION = 0.3
__C.AUGMENTATION.COLOR_JITTER.HUE = 0.2

__C.AUGMENTATION.USE_GASSIAN_BLUR = edict()
__C.AUGMENTATION.USE_GASSIAN_BLUR.PROB = 0.5

__C.AUGMENTATION.RANDOM_ROTATE = edict()
__C.AUGMENTATION.RANDOM_ROTATE.PROB = 0.5
__C.AUGMENTATION.RANDOM_ROTATE.DEGREE = 20


__C.AUGMENTATION.RANDOM_RESIZE_CROP = edict()
__C.AUGMENTATION.RANDOM_RESIZE_CROP.SCALE = (0.5, 1)
__C.AUGMENTATION.RANDOM_RESIZE_CROP.RATIO = (3./4., 4./3.)

__C.AUGMENTATION.RANDOM_HORIZONTAL_FLIP = edict()
__C.AUGMENTATION.RANDOM_HORIZONTAL_FLIP.PROB = 0.5

__C.AUGMENTATION.CUTOUT = edict()
__C.AUGMENTATION.CUTOUT.PROB = 1.0
__C.AUGMENTATION.CUTOUT.MASK_SIZE = 16
__C.AUGMENTATION.CUTOUT.CUT_INSIDE = False
__C.AUGMENTATION.CUTOUT.MASK_COLOR = 0
__C.AUGMENTATION.CUTOUT.DUAL_CUTOUT_ALPHA = 0.1

__C.AUGMENTATION.RANDOM_ERASING = edict()
__C.AUGMENTATION.RANDOM_ERASING.PROB = 0.5
__C.AUGMENTATION.RANDOM_ERASING.AREA_RATIO_RANGE = [0.02, 0.4]
__C.AUGMENTATION.RANDOM_ERASING.MIN_ASPECT_RATIO = 0.3
__C.AUGMENTATION.RANDOM_ERASING.MAX_ATTEMPT = 20

__C.AUGMENTATION.MIXUP = edict()
__C.AUGMENTATION.MIXUP.ALPHA = 1.0

__C.AUGMENTATION.RICAP = edict()
__C.AUGMENTATION.RICAP.BETA = 0.3

__C.AUGMENTATION.CUTMIX = edict()
__C.AUGMENTATION.CUTMIX.ALPHA = 1.0

__C.AUGMENTATION.LABEL_SMOOTHING = edict()
__C.AUGMENTATION.LABEL_SMOOTHING.EPSILON = 0.1

__C.AUGMENTATION.GRIDMASK = edict()
__C.AUGMENTATION.GRIDMASK.D1 = 10
__C.AUGMENTATION.GRIDMASK.D2 = 20
__C.AUGMENTATION.GRIDMASK.ROTATE = 1
__C.AUGMENTATION.GRIDMASK.RATIO = 0.5
__C.AUGMENTATION.GRIDMASK.MODE = 0
__C.AUGMENTATION.GRIDMASK.PROB = 1.


# scheduler
__C.SCHEDULER = edict()
__C.SCHEDULER.EPOCHS = 160                                                       # 学习率衰减步数
# warm up (options: none, linear, exponential)
__C.SCHEDULER.WARMUP = edict()
__C.SCHEDULER.WARMUP.TYPE = 'none'
__C.SCHEDULER.WARMUP.EPOCHS = 0                                                  # WARMUP步数
__C.SCHEDULER.WARMUP.START_FACTOR = 1e-3
__C.SCHEDULER.WARMUP.EXPONENT = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
__C.SCHEDULER.TYPE = 'constant'
__C.SCHEDULER.MILESTONES = [80, 120]
__C.SCHEDULER.LR_DECAY = 0.1
__C.SCHEDULER.LR_MIN_FACTOR = 0.001
__C.SCHEDULER.T0 = 10
__C.SCHEDULER.T_MUL = 1.

__C.TTA = edict()
__C.TTA.USE_RESIZE = False
__C.TTA.USE_CENTER_CROP = False
__C.TTA.RESIZE = 256






__C.OPTIM = edict()
# Adam
__C.OPTIM.ADAM = edict()
__C.OPTIM.ADAM.BETAS = (0.9, 0.999)
# LARS
__C.OPTIM.LARS = edict()
__C.OPTIM.LARS.EPS = 1e-9
__C.OPTIM.LARS.THRESHOLD = 1e-2
# AdaBound
__C.OPTIM.ADABOUND = edict()
__C.OPTIM.ADABOUND.BETAS = (0.9, 0.999)
__C.OPTIM.ADABOUND.FINAL_LR = 0.1
__C.OPTIM.ADABOUND.GAMMA = 1e-3





# region 模型配置


# endregion

# region 数据集配置


# endregion


# 使用gpu的数目
__C.GPUS = 1


__C.WEIGHT_DECAY = 5e-4
__C.MOMENTUM = 0.9
# 初始学习率
__C.LR = 1e-3

# 训练好模型的保存位置


# 训练完成，权重文件的保存路径,默认保存在trained_model下
__C.TRAINED_MODEL = './weights/resnext101_32x32d/epoch_40.pth'

__C.MODEL_NAMES = {
    'alexnet': Basenet,
    'resnext50_32x4d': Basenet,
    'resnext101_32x8d': Basenet,
    'resnext101_32x8d_wsl': Basenet,
    'resnext101_32x16d_wsl': Basenet,
    'resnext101_32x48d_wsl': Basenet,
    'resnext101_32x32d_wsl': Basenet,
    'resnet18': Basenet,
    'resnet34': Basenet,
    'resnet50': Basenet,
    'resnet101': Basenet,
    'resnet152': Basenet,
    'moblienet_v2': Basenet,
    'googlenet': Basenet,
    'inception_v3': Basenet,
    'shufflenet_v2_x0_5': Basenet,
    'shufflenet_v2_x1_0': Basenet,
    'shufflenet_v2_x1_5': Basenet,
    'shufflenet_v2_x2_0': Basenet,
    'squeezenet1_0': Basenet,
    'squeezenet1_1': Basenet,
    'vgg11': Basenet,
    'vgg16': Basenet,
    'vgg13': Basenet,
    'vgg19': Basenet,
    'vgg11_bn': Basenet,
    'vgg16_bn': Basenet,
    'vgg13_bn': Basenet,
    'vgg19_bn': Basenet,
    'densenet121': Densenet,
    'densenet161': Densenet,
    'densenet169': Densenet,
    'densenet1201': Densenet,
    'efficientnet-b0': Efficientnet,
    'efficientnet-b1': Efficientnet,
    'efficientnet-b2': Efficientnet,
    'efficientnet-b3': Efficientnet,
    'efficientnet-b4': Efficientnet,
    'efficientnet-b5': Efficientnet,
    'efficientnet-b6': Efficientnet,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet,
    'efficientnet-l2': Efficientnet
}
