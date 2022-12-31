# 教程1：了解配置文件

我们将模块化和继承性设计融入到我们的配置文件系统中，方便进行各种实验。如果您想查看配置文件，你可以运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置文件。你也可以通过传递参数 `--cfg-options xxx.yyy=zzz` 来查看更新的配置信息。

## 配置文件的结构

在  `config/_base_ ` 文件夹下面有4种基本组件类型： 数据集(dataset)，模型(model)，训练策略(schedule)和运行时的默认设置(default runtime)。许多模型都可以很容易地通过组合这些组件进行实现，比如 DeepLabV3，PSPNet。使用 `_base_` 下的组件构建的配置信息叫做原始配置 (primitive)。

对于同一个文件夹下的所有配置文件，建议**只有一个**对应的**原始配置文件**。所有其他的配置文件都应该继承自这个原始配置文件，从而保证每个配置文件的最大继承深度为 3。

为了便于理解，我们建议社区贡献者从现有的方法继承。例如，如果您在 DeepLabV3 基础上进行了一些修改，用户可以先通过指定 `_base_ = ../deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py` 继承基本的 DeepLabV3 结构，然后在配置文件中修改必要的字段。

如果你正在构建一个全新的方法，它不与现有的任何方法共享基本组件，您可以在`config`下创建一个新的文件夹`xxxnet` ，详细文档请参考[mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/config.html)。

## 配置文件命名风格

我们遵循以下格式来命名配置文件，建议社区贡献者遵循相同的风格。

```text
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}
```

配置文件的文件名分为五个部分，组成文件名每一个部分和组件之间都用`_`连接，每个部分或组件中的每个单词都要用`-`连接。

- `{algorithm name}`: 算法的名称，如 `deeplabv3`,  `pspnet` 等。
- `{model component names}`: 算法中使用的组件名称，如主干(backbone)、解码头(head)等。例如，`r50-d8 `表示使用ResNet50主干网络，并使用主干网络的8倍下采样输出作为下一级的输入。
- `{training settings}`: 训练时的参数设置，如 `batch size`、数据增强(augmentation)、损失函数(loss)、学习率调度器(learning rate scheduler)和训练轮数(epochs/iterations)。例如: `4xb4-ce-linearlr-40K` 意味着使用4个gpu，每个gpu4个图像，使用交叉熵损失函数(CrossEntropy)，线性学习率调度程序，训练40K iterations。
  一些缩写:
  - `{gpu x batch_per_gpu}`: GPU数量和每个GPU的样本数。`bN ` 表示每个GPU的batch size为N，如 `8xb2` 为8个gpu x 每个gpu2张图像的缩写。如果未提及，则默认使用 `4xb4 `。
  - `{schedule}`: 训练计划，选项有`20k`，`40k`等。`20k ` 和 `40k` 分别表示20000次迭代(iterations)和40000次迭代(iterations)。
- `{training dataset information}`: 训练数据集名称，如 `cityscapes `， `ade20k ` 等，以及输入分辨率。例如: `cityscapes-768x768  `表示使用 `cityscapes` 数据集进行训练，输入分辨率为`768x768 `。
- `{testing dataset information}` (可选): 测试数据集名称。当您的模型在一个数据集上训练但在另一个数据集上测试时，请将测试数据集名称添加到此处。如果没有这一部分，则意味着模型是在同一个数据集上进行训练和测试的。

## PSPNet 的一个例子

为了帮助用户熟悉对这个现代语义分割系统的完整配置文件和模块，我们对使用ResNet50V1c作为主干网络的PSPNet的配置文件作如下的简要注释和说明。要了解更详细的用法和每个模块对应的替换方法，请参阅API文档。

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
] # 我们可以在基本配置文件的基础上 构建新的配置文件
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
```

`_base_/models/pspnet_r50-d8.py`是使用ResNet50V1c作为主干网络的PSPNet的基本模型配置文件。

```python
# 模型设置
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 分割框架通常使用 SyncBN
data_preprocessor = dict(  # 数据预处理的配置项，通常包括图像的归一化和增强
    type='SegDataPreProcessor',  # 数据预处理的类型
    mean=[123.675, 116.28, 103.53],  # 用于归一化输入图像的平均值
    std=[58.395, 57.12, 57.375],  # 用于归一化输入图像的标准差
    bgr_to_rgb=True,  # 是否将图像从 BGR 转为 RGB
    pad_val=0,  # 图像的填充值
    seg_pad_val=255)  # 'gt_seg_map'的填充值
model = dict(
    type='EncoderDecoder',  # 分割器(segmentor)的名字
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',  # 加载使用 ImageNet 预训练的主干网络
    backbone=dict(
        type='ResNetV1c',  # 主干网络的类别，更多细节请参考 mmseg/models/backbones/resnet.py
        depth=50,  # 主干网络的深度，通常为 50 和 101
        num_stages=4,  # 主干网络状态(stages)的数目
        out_indices=(0, 1, 2, 3),  # 每个状态(stage)产生的特征图输出的索引
        dilations=(1, 1, 2, 4),  # 每一层(layer)的空心率(dilation rate)
        strides=(1, 2, 1, 1),  # 每一层(layer)的步长(stride)
        norm_cfg=norm_cfg,  # 归一化层(norm layer)的配置项
        norm_eval=False,  # 是否冻结 BN 里的统计项
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积
        contract_dilation=True),  # 当空洞率 > 1, 是否压缩第一个空洞层
    decode_head=dict(
        type='PSPHead',  # 解码头(decode head)的类别。可用选项请参 mmseg/models/decode_heads
        in_channels=2048,  # 解码头的输入通道数
        in_index=3,  # 被选择特征图(feature map)的索引
        channels=512,  # 解码头中间态(intermediate)的通道数
        pool_scales=(1, 2, 3, 6),  # PSPHead 平均池化(avg pooling)的规模(scales)。 细节请参考文章内容
        dropout_ratio=0.1,  # 进入最后分类层(classification layer)之前的 dropout 比例
        num_classes=19,  # 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150
        norm_cfg=norm_cfg,  # 归一化层的配置项
        align_corners=False,  # 解码过程中调整大小(resize)的 align_corners 参数
        loss_decode=dict(  # 解码头(decode_head)里的损失函数的配置项
            type='CrossEntropyLoss',  # 分割时使用的损失函数的类别
            use_sigmoid=False,  # 分割时是否使用 sigmoid 激活
            loss_weight=1.0)),  # 解码头的损失权重
    auxiliary_head=dict(
        type='FCNHead',  # 辅助头(auxiliary head)的种类。可用选项请参考 mmseg/models/decode_heads
        in_channels=1024,  # 辅助头的输入通道数
        in_index=2,  # 被选择的特征图(feature map)的索引
        channels=256,  # 辅助头中间态(intermediate)的通道数
        num_convs=1,  # FCNHead 里卷积(convs)的数目，辅助头中通常为1
        concat_input=False,  # 在分类层(classification layer)之前是否连接(concat)输入和卷积的输出
        dropout_ratio=0.1,  # 进入最后分类层(classification layer)之前的 dropout 比例
        num_classes=19,  # 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150
        norm_cfg=norm_cfg,  # 归一化层的配置项
        align_corners=False,  # 解码过程中调整大小(resize)的 align_corners 参数
        loss_decode=dict(  # 辅助头(auxiliary head)里的损失函数的配置项
            type='CrossEntropyLoss',  # 分割时使用的损失函数的类别
            use_sigmoid=False,  # 分割时是否使用 sigmoid 激活
            loss_weight=0.4)),  # 辅助头损失的权重，默认设置为0.4
    # 模型训练和测试设置项
    train_cfg=dict(),  # train_cfg 当前仅是一个占位符
    test_cfg=dict(mode='whole'))  # 测试模式，可选参数为 'whole' 和 'slide'. 'whole': 在整张图像上全卷积(fully-convolutional)测试。 'slide': 在输入图像上做滑窗预测
```

`_base_/datasets/cityscapes.py`是数据集的基本配置文件。

```python
# 数据集设置
dataset_type = 'CityscapesDataset'  # 数据集类型，这将被用来定义数据集
data_root = 'data/cityscapes/'  # 数据的根路径
crop_size = (512, 1024)  # 训练时的裁剪大小
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations'),  # 第2个流程，对于当前图像，加载它的标注图像
    dict(type='RandomResize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
        scale=(2048, 1024),  # 图像裁剪的大小
        ratio_range=(0.5, 2.0),  # 数据增广的比例范围
        keep_ratio=True),  # 调整图像大小时是否保持纵横比
    dict(type='RandomCrop',  # 随机裁剪当前图像和其标注图像的数据增广流程
        crop_size=crop_size,  # 随机裁剪的大小
        cat_max_ratio=0.75),  # 单个类别可以填充的最大区域的比
    dict(type='RandomFlip',  # 翻转图像和其标注图像的数据增广流程
        prob=0.5),  # 翻转图像的概率
    dict(type='PhotoMetricDistortion'),  # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程
    dict(type='PackSegInputs')  # 打包用于语义分割的输入数据
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像
    dict(type='Resize',  # 使用调整图像大小(resize)增强
        scale=(2048, 1024),  # 图像缩放的大小
        keep_ratio=True),  # 在调整图像大小时是否保留长宽比
    # 在' Resize '之后添加标注图像
    # 不需要做调整图像大小(resize)的数据变换
    dict(type='LoadAnnotations'),  # 加载数据集提供的语义分割标注
    dict(type='PackSegInputs')  # 打包用于语义分割的输入数据
]
train_dataloader = dict(  # 训练数据加载器(dataloader)的配置
    batch_size=2,  # 每一个GPU的batch size大小
    num_workers=2,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 训练时进行随机洗牌(shuffle)
    dataset=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),  # 训练数据的前缀
        pipeline=train_pipeline)) # 数据处理流程，它通过之前创建的train_pipeline传递。
val_dataloader = dict(
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),  # 测试数据的前缀
        pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。
test_dataloader = val_dataloader
# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

`_base_/schedules/schedule_40k.py`

```python
# optimizer
optimizer = dict(type='SGD', # 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py
                lr=0.01,  # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档
                momentum=0.9,  # 动量大小 (Momentum)
                weight_decay=0.0005)  # SGD 的权重衰减 (weight decay)
optim_wrapper = dict(type='OptimWrapper',  # 优化器包装器(Optimizer wrapper)为更新参数提供了一个公共接口
                    optimizer=optimizer,  # 用于更新模型参数的优化器(Optimizer)
                    clip_grad=None)  # 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数。
# 学习策略
param_scheduler = [
    dict(
        type='PolyLR',  # 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等. 请从 https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py 参考 LrUpdater 的细节
        eta_min=1e-4,  # 训练结束时的最小学习率
        power=0.9,  # 多项式衰减 (polynomial decay) 的幂
        begin=0,  # 开始更新参数的时间步(step)
        end=40000,  # 停止更新参数的时间步(step)
        by_epoch=False)  # 是否按照 epoch 计算训练时间
]
# 40k iteration 的训练计划
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# 默认钩子(hook)配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 记录迭代过程中花费的时间
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # 从'Runner'的不同组件收集和写入日志
    param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中的一些超参数，例如学习率
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),  # 定期保存检查点(checkpoint)
    sampler_seed=dict(type='DistSamplerSeedHook'))  # 用于分布式训练的数据加载采样器
```

in `_base_/default_runtime.py`

```python
# 将注册表的默认范围设置为mmseg
default_scope = 'mmseg'
# environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
log_processor = dict(by_epoch=False)
load_from = None  # 从文件中加载检查点(checkpoint)
resume = False  # 是否从已有的模型恢复
```

这些都是用于训练和测试PSPNet的配置文件，要加载和解析它们，我们可以使用[MMEngine](https://github.com/open-mmlab/mmengine)实现的[Config](https://mmengine.readthedocs.io/en/latest/tutorials/config.html)。

```python
from mmengine.config import Config

cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')
print(cfg.train_dataloader)
```

```shell
{'batch_size': 2,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'type': 'InfiniteSampler', 'shuffle': True},
 'dataset': {'type': 'CityscapesDataset',
  'data_root': 'data/cityscapes/',
  'data_prefix': {'img_path': 'leftImg8bit/train',
   'seg_map_path': 'gtFine/train'},
  'pipeline': [{'type': 'LoadImageFromFile'},
   {'type': 'LoadAnnotations'},
   {'type': 'RandomResize',
    'scale': (2048, 1024),
    'ratio_range': (0.5, 2.0),
    'keep_ratio': True},
   {'type': 'RandomCrop', 'crop_size': (512, 1024), 'cat_max_ratio': 0.75},
   {'type': 'RandomFlip', 'prob': 0.5},
   {'type': 'PhotoMetricDistortion'},
   {'type': 'PackSegInputs'}]}}
```

`cfg `是`mmengine.config.Config `的一个实例。它的接口与dict对象相同，也允许将配置值作为属性访问。更多信息请参见[MMEngine](https://github.com/open-mmlab/mmengine)中的[config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html)。

## FAQ

### 忽略基础配置文件里的一些字段

有时，您可以设置`_delete_=True `来忽略基本配置文件中的某些字段。您可以参考[MMEngine](https://github.com/open-mmlab/mmengine)中的[config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html)来获得一些简单的指导。

例如，在MMSegmentation中，如果您想在下面的配置文件`pspnet.py `中修改PSPNet的主干网络:

```python
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
```

用以下代码加载并解析配置文件`pspnet.py`:

```python
from mmengine.config import Config

cfg = Config.fromfile('pspnet.py')
print(cfg.model)
```

```shell
{'type': 'EncoderDecoder',
 'pretrained': 'torchvision://resnet50',
 'backbone': {'type': 'ResNetV1c',
  'depth': 50,
  'num_stages': 4,
  'out_indices': (0, 1, 2, 3),
  'dilations': (1, 1, 2, 4),
  'strides': (1, 2, 1, 1),
  'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
  'norm_eval': False,
  'style': 'pytorch',
  'contract_dilation': True},
 'decode_head': {'type': 'PSPHead',
  'in_channels': 2048,
  'in_index': 3,
  'channels': 512,
  'pool_scales': (1, 2, 3, 6),
  'dropout_ratio': 0.1,
  'num_classes': 19,
  'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
  'align_corners': False,
  'loss_decode': {'type': 'CrossEntropyLoss',
   'use_sigmoid': False,
   'loss_weight': 1.0}}}
```

`ResNet`和`HRNet`使用不同的关键字构建，编写一个新的配置文件`hrnet.py`，如下所示:

```python
_base_ = 'pspnet.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        norm_cfg=norm_cfg,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))))
```

用以下代码加载并解析配置文件`hrnet.py`:

```python
from mmengine.config import Config
cfg = Config.fromfile('hrnet.py')
print(cfg.model)
```

```shell
{'type': 'EncoderDecoder',
 'pretrained': 'open-mmlab://msra/hrnetv2_w32',
 'backbone': {'type': 'HRNet',
  'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
  'extra': {'stage1': {'num_modules': 1,
    'num_branches': 1,
    'block': 'BOTTLENECK',
    'num_blocks': (4,),
    'num_channels': (64,)},
   'stage2': {'num_modules': 1,
    'num_branches': 2,
    'block': 'BASIC',
    'num_blocks': (4, 4),
    'num_channels': (32, 64)},
   'stage3': {'num_modules': 4,
    'num_branches': 3,
    'block': 'BASIC',
    'num_blocks': (4, 4, 4),
    'num_channels': (32, 64, 128)},
   'stage4': {'num_modules': 3,
    'num_branches': 4,
    'block': 'BASIC',
    'num_blocks': (4, 4, 4, 4),
    'num_channels': (32, 64, 128, 256)}}},
 'decode_head': {'type': 'PSPHead',
  'in_channels': 2048,
  'in_index': 3,
  'channels': 512,
  'pool_scales': (1, 2, 3, 6),
  'dropout_ratio': 0.1,
  'num_classes': 19,
  'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
  'align_corners': False,
  'loss_decode': {'type': 'CrossEntropyLoss',
   'use_sigmoid': False,
   'loss_weight': 1.0}}}
```

`_delete_=True` 将用新的键去替换 `backbone` 字段内所有旧的键。

### 使用配置文件里的中间变量

配置文件中会使用一些中间变量，例如数据集(datasets)字段里的 `train_pipeline`/`test_pipeline`。 需要注意的是，在子配置文件里修改中间变量时，您需要再次传递这些变量给对应的字段。例如，我们想改变在训练或测试PSPNet时采用的多尺度策略 (multi scale strategy)，`train_pipeline`/`test_pipeline` 是我们需要修改的中间变量。

```python
_base_ = '../pspnet/pspnet_r50-d8_4xb4-40k_cityscpaes-512x1024.py'
crop_size = (512, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize',
         img_scale=(2048, 1024),
         ratio_range=(1., 2.),
         keep_ration=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',
        scale=(2048, 1024),
        keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline)
test_dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline)
train_dataloader = dict(dataset=train_dataset)
val_dataloader = dict(dataset=test_dataset)
test_dataloader = val_dataloader
```

我们首先需要定义新的 `train_pipeline`/`test_pipeline` 然后传递到 `dataset` 里。

类似的，如果我们想从 `SyncBN` 切换到 `BN` 或者 `MMSyncBN`，我们需要替换配置文件里的每一个 `norm_cfg`。

```python
_base_ = '../pspnet/pspnet_r50-d8_4xb4-40k_cityscpaes-512x1024.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
```

## 通过脚本参数修改配置文件

在[training script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/train.py)和[testing script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/test.py)中，我们支持脚本参数 `--cfg-options`，它可以帮助用户覆盖所使用的配置中的一些设置，`xxx=yyy` 格式的键值对将合并到配置文件中。

例如，这是一个简化的脚本 `demo_script.py `:

```python
import argparse

from mmengine.config import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='Script Example')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print(cfg)

if __name__ == '__main__':
    main()
```

一个配置文件示例 `demo_config.py` 如下所示:

```python
backbone = dict(
    type='ResNetV1c',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    dilations=(1, 1, 2, 4),
    strides=(1, 2, 1, 1),
    norm_eval=False,
    style='pytorch',
    contract_dilation=True)
```

运行 `demo_script.py`:

```shell
python demo_script.py demo_config.py
```

```shell
Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
```

通过脚本参数修改配置:

```shell
python demo_script.py demo_config.py --cfg-options backbone.depth=101
```

```shell
Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 101, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
```

- 更新列表/元组的值。

  如果要更新的值是一个 list 或 tuple。例如，需要在配置文件 `demo_config.py ` 的 `backbone ` 中设置 `stride =(1,2,1,1) `。
  如果您想更改这个键，你可以用两种方式进行指定:

  1. `--cfg-options backbone.strides="(1, 1, 1, 1)"`. 注意引号 " 是支持 list/tuple 数据类型所必需的。

     ```shell
     python demo_script.py demo_config.py --cfg-options backbone.strides="(1, 1, 1, 1)"
     ```

     ```shell
     Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 1, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
     ```

  2. `--cfg-options backbone.strides=1,1,1,1`. 注意，在指定的值中**不允许**有空格。

     另外，如果原来的类型是tuple，通过这种方式修改后会自动转换为list。

     ```shell
     python demo_script.py demo_config.py --cfg-options backbone.strides=1,1,1,1
     ```

     ```shell
     Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': [1, 1, 1, 1], 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
     ```

```{note}
  这种修改方法仅支持修改string、int、float、boolean、None、list和tuple类型的配置项。
  具体来说，对于list和tuple类型的配置项，它们内部的元素也必须是上述七种类型之一。
```
