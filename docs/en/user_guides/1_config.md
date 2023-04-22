# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.
You may also pass `--cfg-options xxx.yyy=zzz` to see updated config.

## Config File Structure

There are 4 basic component types under `config/_base_`, datasets, models, schedules, default_runtime.
Many methods could be easily constructed with one of each like DeepLabV3, PSPNet.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on DeepLabV3, user may first inherit the basic DeepLabV3 structure by specifying `_base_ = ../deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxxnet` under `configs`,

Please refer to [mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```text
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}
```

The file name is divided to five parts. All parts and components are connected with `_` and words of each part or component should be connected with `-`.

- `{algorithm name}`: The name of the algorithm, such as `deeplabv3`, `pspnet`, etc.
- `{model component names}`: Names of the components used in the algorithm such as backbone, head, etc. For example, `r50-d8` means using ResNet50 backbone and use output of backbone is 8 times downsampling as input.
- `{training settings}`: Information of training settings such as batch size, augmentations, loss, learning rate scheduler, and epochs/iterations. For example: `4xb4-ce-linearlr-40K` means using 4-gpus x 4-images-per-gpu, CrossEntropy loss, Linear learning rate scheduler, and train 40K iterations.
  Some abbreviations:
  - `{gpu x batch_per_gpu}`: GPUs and samples per GPU. `bN` indicates N batch size per GPU. E.g. `8xb2` is the short term of 8-gpus x 2-images-per-gpu. And `4xb4` is used by default if not mentioned.
  - `{schedule}`: training schedule, options are `20k`, `40k`, etc. `20k` and `40k` means 20000 iterations and 40000 iterations respectively.
- `{training dataset information}`: Training dataset names like `cityscapes`, `ade20k`, etc, and input resolutions. For example: `cityscapes-768x768` means training on `cityscapes` dataset and the input shape is `768x768`.
- `{testing dataset information}` (optional): Testing dataset name for models trained on one dataset but tested on another. If not mentioned, it means the model was trained and tested on the same dataset type.

## An Example of PSPNet

To help the users have a basic idea of a complete config and the modules in a modern semantic segmentation system,
we make brief comments on the config of PSPNet using ResNet50V1c as the following.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation.

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
] # base config file which we build new config file on.
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
```

`_base_/models/pspnet_r50-d8.py` is a basic model cfg file for PSPNet using ResNet50V1c

```python
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN
data_preprocessor = dict(  # The config of data preprocessor, usually includes image normalization and augmentation.
    type='SegDataPreProcessor',  # The type of data preprocessor.
    mean=[123.675, 116.28, 103.53],  # Mean values used for normalizing the input images.
    std=[58.395, 57.12, 57.375],  # Standard variance used for normalizing the input images.
    bgr_to_rgb=True,  # Whether to convert image from BGR to RGB.
    pad_val=0,  # Padding value of image.
    seg_pad_val=255)  # Padding value of segmentation map.
model = dict(
    type='EncoderDecoder',  # Name of segmentor
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',  # The ImageNet pretrained backbone to be loaded
    backbone=dict(
        type='ResNetV1c',  # The type of backbone. Please refer to mmseg/models/backbones/resnet.py for details.
        depth=50,  # Depth of backbone. Normally 50, 101 are used.
        num_stages=4,  # Number of stages of backbone.
        out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stages.
        dilations=(1, 1, 2, 4),  # The dilation rate of each layer.
        strides=(1, 2, 1, 1),  # The stride of each layer.
        norm_cfg=norm_cfg,  # The configuration of norm layer.
        norm_eval=False,  # Whether to freeze the statistics in BN
        style='pytorch',  # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.
        contract_dilation=True),  # When dilation > 1, whether contract first layer of dilation.
    decode_head=dict(
        type='PSPHead',  # Type of decode head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=2048,  # Input channel of decode head.
        in_index=3,  # The index of feature map to select.
        channels=512,  # The intermediate channels of decode head.
        pool_scales=(1, 2, 3, 6),  # The avg pooling scales of PSPHead. Please refer to paper for details.
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=19,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=norm_cfg,  # The configuration of norm layer.
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Config of loss function for the decode_head.
            type='CrossEntropyLoss',  # Type of loss used for segmentation.
            use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
            loss_weight=1.0)),  # Loss weight of decode_head.
    auxiliary_head=dict(
        type='FCNHead',  # Type of auxiliary head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=1024,  # Input channel of auxiliary head.
        in_index=2,  # The index of feature map to select.
        channels=256,  # The intermediate channels of decode head.
        num_convs=1,  # Number of convs in FCNHead. It is usually 1 in auxiliary head.
        concat_input=False,  # Whether concat output of convs with input before classification layer.
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=19,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=norm_cfg,  # The configuration of norm layer.
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Config of loss function for the auxiliary_head.
            type='CrossEntropyLoss',  # Type of loss used for segmentation.
            use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
            loss_weight=0.4)),  # Loss weight of auxiliary_head.
    # model training and testing settings
    train_cfg=dict(),  # train_cfg is just a place holder for now.
    test_cfg=dict(mode='whole'))  # The test mode, options are 'whole' and 'slide'. 'whole': whole image fully-convolutional test. 'slide': sliding crop window on the image.
```

`_base_/datasets/cityscapes.py` is the configuration file of the dataset

```python
# dataset settings
dataset_type = 'CityscapesDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/cityscapes/'  # Root path of data.
crop_size = (512, 1024)  # The crop size during training.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
    dict(type='RandomResize',  # Augmentation pipeline that resize the images and their annotations.
        scale=(2048, 1024),  # The scale of image.
        ratio_range=(0.5, 2.0),  # The augmented scale range as ratio.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=crop_size,  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        prob=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize',  # Use resize augmentation
        scale=(2048, 1024),  # Images scales for resizing.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),  # Load annotations for semantic segmentation provided by dataset.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
train_dataloader = dict(  # Train dataloader config
    batch_size=2,  # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),  # Randomly shuffle during training.
    dataset=dict(  # Train dataset config
        type=dataset_type,  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,  # The root of dataset.
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),  # Prefix for training data.
        pipeline=train_pipeline)) # Processing pipeline. This is passed by the train_pipeline created before.
val_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU
    num_workers=4,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate testing speed.
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(  # Test dataset config
        type=dataset_type,  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,  # The root of dataset.
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),  # Prefix for testing data.
        pipeline=test_pipeline))  # Processing pipeline. This is passed by the test_pipeline created before.
test_dataloader = val_dataloader
# The metric to measure the accuracy. Here, we use IoUMetric.
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

`_base_/schedules/schedule_40k.py`

```python
# optimizer
optimizer = dict(type='SGD', # Type of optimizers, refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py for more details
                lr=0.01,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
                momentum=0.9,  # Momentum
                weight_decay=0.0005)  # Weight decay of SGD
optim_wrapper = dict(type='OptimWrapper',  # Optimizer wrapper provides a common interface for updating parameters.
                    optimizer=optimizer,  # Optimizer used to update model parameters.
                    clip_grad=None)  # If ``clip_grad`` is not None, it will be the arguments of ``torch.nn.utils.clip_grad``.
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',  # The policy of scheduler, also support Step, CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py
        eta_min=1e-4,  # Minimum learning rate at the end of scheduling.
        power=0.9,  # The power of polynomial decay.
        begin=0,  # Step at which to start updating the parameters.
        end=40000,  # Step at which to stop updating the parameters.
        by_epoch=False)  # Whether count by epoch or not.
]
# training schedule for 40k iteration
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Log the time spent during iteration.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # Collect and write logs from different components of ``Runner``.
    param_scheduler=dict(type='ParamSchedulerHook'),  # update some hyper-parameters in optimizer, e.g., learning rate.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),  # Save checkpoints periodically.
    sampler_seed=dict(type='DistSamplerSeedHook'))  # Data-loading sampler for distributed training.
```

in `_base_/default_runtime.py`

```python
# Set the default scope of the registry to mmseg.
default_scope = 'mmseg'
# environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
log_processor = dict(by_epoch=False)
load_from = None  # Load checkpoint from file.
resume = False  # Whether to resume from existed model.
```

These are all the configs for training and testing PSPNet, to load and parse them, we can use [Config](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) implemented in [MMEngine](https://github.com/open-mmlab/mmengine)

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

`cfg` is an instance of `mmengine.config.Config`, its interface is the same as a dict object and also allows access config values as attributes. See [config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for more information.

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of the fields in base configs.
See [config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for simple illustration.

In MMSegmentation, for example, if you would like to modify the backbone of PSPNet with the following config file `pspnet.py`:

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

Load and parse the config file `pspnet.py` in the code as follows:

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

`ResNet` and `HRNet` use different keywords to construct, write a new config file `hrnet.py` as follows:

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

Load and parse the config file `hrnet.py` in the code as follows:

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

The `_delete_=True` would replace all old keys in `backbone` field with new keys.

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, user need to pass the intermediate variables into corresponding fields again.
For example, we would like to change multi scale strategy to train/test a PSPNet. `train_pipeline`/`test_pipeline` are intermediate variable we would like to modify.

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

We first define the new `train_pipeline`/`test_pipeline` and pass them into `dataset`.

Similarly, if we would like to switch from `SyncBN` to `BN` or `MMSyncBN`, we need to substitute every `norm_cfg` in the config.

```python
_base_ = '../pspnet/pspnet_r50-d8_4xb4-40k_cityscpaes-512x1024.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
```

## Modify config through script arguments

In the [training script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/train.py) and the [testing script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/test.py), we support the script argument `--cfg-options`, it may help users override some settings in the used config, the key-value pair in `xxx=yyy` format will be merged into config file.

For example, this is a simplified script `demo_script.py`:

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

An example config file `demo_config.py` as follows:

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

Run `demo_script.py`:

```shell
python demo_script.py demo_config.py
```

```shell
Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
```

Modify config through script arguments:

```shell
python demo_script.py demo_config.py --cfg-options backbone.depth=101
```

```shell
Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 101, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
```

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file `demo_config.py` sets `strides=(1, 2, 1, 1)` in `backbone`.
  If you want to change this key, you may specify in two ways:

  1. `--cfg-options backbone.strides="(1, 1, 1, 1)"`. Note that the quotation mark " is necessary to support list/tuple data types.

     ```shell
     python demo_script.py demo_config.py --cfg-options backbone.strides="(1, 1, 1, 1)"
     ```

     ```shell
     Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 1, 1, 1), 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
     ```

  2. `--cfg-options backbone.strides=1,1,1,1`. Note that **NO** white space is allowed in the specified value.
     In addition, if the original type is tuple, it will be automatically converted to list after this way.

     ```shell
     python demo_script.py demo_config.py --cfg-options backbone.strides=1,1,1,1
     ```

     ```shell
     Config (path: demo_config.py): {'backbone': {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': [1, 1, 1, 1], 'norm_eval': False, 'style': 'pytorch', 'contract_dilation': True}}
     ```

```{note}
    This modification method only supports modifying configuration items of string, int, float, boolean, None, list and tuple types.
    More specifically, for list and tuple types, the elements inside them must also be one of the above seven types.
```
