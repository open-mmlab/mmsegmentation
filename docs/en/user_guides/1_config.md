# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/print_config.py /PATH/TO/CONFIG` to see the complete config.
You may also pass `--cfg-options xxx.yyy=zzz` to see updated config.

## Config File Structure

There are 4 basic component types under `config/_base_`, dataset, model, schedule, default_runtime.
Many methods could be easily constructed with one of each like DeepLabV3, PSPNet.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on DeepLabV3, user may first inherit the basic DeepLabV3 structure by specifying `_base_ = ../deeplabv3/deeplabv3_r50_512x1024_40ki_cityscapes.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxxnet` under `configs`,

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_{backbone}_[misc]_[gpu x batch_per_gpu]_{schedule}_{training datasets}_[input_size].py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `psp`, `deeplabv3`, etc.
- `{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `mstrain`.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
- `{schedule}`: number of training iterations like `160k`.
- `{training datasets}`: dataset like `cityscapes`, `voc12aug`, `ade`.
- `[input_size]`: the size of training images.

## An Example of PSPNet

To help the users have a basic idea of a complete config and the modules in a modern semantic segmentation system,
we make brief comments on the config of PSPNet using ResNet50V1c as the following.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation.

```python
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN
data_preprocessor = dict(  # The config of data preprocessor, usually includes image normalization and augmentation.
    type='SegDataPreProcessor',  # The type of data preprocessor.
    mean=[123.675, 116.28, 103.53],  # Mean values used for normalizing the input images.
    std=[58.395, 57.12, 57.375],  # Standard variance used for normalizing the input images.
    bgr_to_rgb=True,  # Whether to convert image from BGR to RGB.
    pad_val=0,  # Padding value.
    seg_pad_val=255)  # Padding value of segmentation map.
model = dict(
    type='EncoderDecoder',  # Name of segmentor
    pretrained='open-mmlab://resnet50_v1c',  # The ImageNet pretrained backbone to be loaded
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNetV1c',  # The type of backbone. Please refer to mmseg/models/backbones/resnet.py for details.
        depth=50,  # Depth of backbone. Normally 50, 101 are used.
        num_stages=4,  # Number of stages of backbone.
        out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stages.
        dilations=(1, 1, 2, 4),  # The dilation rate of each layer.
        strides=(1, 2, 1, 1),  # The stride of each layer.
        norm_cfg=dict(  # The configuration of norm layer.
            type='SyncBN',  # Type of norm layer. Usually it is SyncBN.
            requires_grad=True),   # Whether to train the gamma and beta in norm
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
        num_classes=21,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # The configuration of norm layer.
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Config of loss function for the decode_head.
            type='CrossEntropyLoss',  # Type of loss used for segmentation.
            use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
            loss_weight=1.0)),  # Loss weight of decode head.
    auxiliary_head=dict(
        type='FCNHead',  # Type of auxiliary head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=1024,  # Input channel of auxiliary head.
        in_index=2,  # The index of feature map to select.
        channels=256,  # The intermediate channels of decode head.
        num_convs=1,  # Number of convs in FCNHead. It is usually 1 in auxiliary head.
        concat_input=False,  # Whether concat output of convs with input before classification layer.
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=21,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # The configuration of norm layer.
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Config of loss function for the decode_head.
            type='CrossEntropyLoss',  # Type of loss used for segmentation.
            use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
            loss_weight=0.4)))  # Loss weight of auxiliary head, which is usually 0.4 of decode head.
dataset_type = 'PascalVOCDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/VOCdevkit/VOC2012'  # Root path of data.
crop_size = (512, 512)  # The crop size during training.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
    dict(type='RandomResize',  # Augmentation pipeline that resize the images and their annotations.
        scale=(2048, 512),  # The largest scale of image.
        ratio_range=(0.5, 2.0),  # The augmented scale range as ratio.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=(512, 512),  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(
        type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        flip_ratio=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    dict(type='Pad',  # Augmentation pipeline that pad the image to specified size.
        size=(512, 512)),  # The output size of padding.
    dict(type='PackSegInputs'),  # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize',  # Use resize augmentation
         scale=(2048, 512),  # Images scales for resizing.
         keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='LoadAnnotations'),  # Load annotations for semantic segmentation provided by dataset.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
dataset_train = dict(  # Train dataset config
    type='PascalVOCDataset',  # Type of dataset, refer to mmseg/datasets/ for details.
    data_root='data/VOCdevkit/VOC2012',  # The root of dataset.
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),  # Prefix for training data.
    ann_file='ImageSets/Segmentation/train.txt', # The annotation file path.
    pipeline=[  # Processing pipeline. This is passed by the train_pipeline created before.
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='RandomResize',
            scale=(2048, 512),
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Pad', size=(512, 512)),
        dict(type='PackSegInputs')
    ])
dataset_aug = dict(  # The augmentation dataset config
    type='PascalVOCDataset',
    data_root='data/VOCdevkit/VOC2012',
    data_prefix=dict(
        img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
    ann_file='ImageSets/Segmentation/aug.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='RandomResize',
            scale=(2048, 512),
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Pad', size=(512, 512)),
        dict(type='PackSegInputs')
    ])
train_dataloader = dict(  # Train dataloader config
    batch_size=4,  # Batch size of a single GPU
    num_workers=4,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),  # Randomly shuffle during training.
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dataset_train,
            dataset_aug
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(
        type='PascalVOCDataset',
        data_root='data/VOCdevkit/VOC2012',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = val_dataloader

# The metric to measure the accuracy. Here, we use IoUMetric.
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
load_from = None
resume = False  # Whether to resume from existed model.

# optimizer
optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005),
    clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,  # The power of polynomial decay.
        begin=0,
        end=40000,
        by_epoch=False)  # Whether count by epoch or not.
]

# Training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(  # config to register logger hook
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),  # Config to set the checkpoint hook
    sampler_seed=dict(type='DistSamplerSeedHook'))

# visualizer
vis_backends = [dict(type='LocalVisBackend')]  # The backend of visualizer.
visualizer = dict(
    type='FlowLocalVisualizer', vis_backends=vis_backends, name='visualizer')

```

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of the fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) for simple illustration.

In MMSegmentation, for example, to change the backbone of PSPNet with the following config.

```python
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MaskRCNN',
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
    decode_head=dict(...),
    auxiliary_head=dict(...))
```

`ResNet` and `HRNet` use different keywords to construct.

```python
_base_ = '../pspnet/psp_r50_512x1024_40ki_cityscpaes.py'
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
                num_channels=(32, 64, 128, 256)))),
    decode_head=dict(...),
    auxiliary_head=dict(...))
```

The `_delete_=True` would replace all old keys in `backbone` field with new keys.

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, user need to pass the intermediate variables into corresponding fields again.
For example, we would like to change multi scale strategy to train/test a PSPNet. `train_pipeline`/`test_pipeline` are intermediate variable we would like to modify.

```python
_base_ = '../pspnet/psp_r50_512x1024_40ki_cityscapes.py'
crop_size = (512, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(1.0, 2.0)),  # change to [1., 2.]
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # change to multi scale testing
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```

We first define the new `train_pipeline`/`test_pipeline` and pass them into `data`.

Similarly, if we would like to switch from `SyncBN` to `BN` or `MMSyncBN`, we need to substitute every `norm_cfg` in the config.

```python
_base_ = '../pspnet/psp_r50_512x1024_40ki_cityscpaes.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
```
