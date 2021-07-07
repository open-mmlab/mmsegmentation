# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/print_config.py /PATH/TO/CONFIG` to see the complete config.
You may also pass `--options xxx.yyy=zzz` to see updated config.

## Config File Structure

There are 4 basic component types under `config/_base_`, dataset, model, schedule, default_runtime.
Many methods could be easily constructed with one of each like DeepLabV3, PSPNet.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from exiting methods.
For example, if some modification is made base on DeepLabV3, user may first inherit the basic DeepLabV3 structure by specifying `_base_ = ../deeplabv3/deeplabv3_r50_512x1024_40ki_cityscapes.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxxnet` under `configs`,

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#config) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_{backbone}_[misc]_[gpu x batch_per_gpu]_{resolution}_{schedule}_{dataset}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `psp`, `deeplabv3`, etc.
- `{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `mstrain`.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
- `{schedule}`: training schedule, `20ki` means 20k iterations.
- `{dataset}`: dataset like `cityscapes`, `voc12aug`, `ade`.

## An Example of PSPNet

To help the users have a basic idea of a complete config and the modules in a modern semantic segmentation system,
we make brief comments on the config of PSPNet using ResNet50V1c as the following.
For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN
model = dict(
    type='EncoderDecoder',  # Name of segmentor
    pretrained='open-mmlab://resnet50_v1c',  # The ImageNet pretrained backbone to be loaded
    backbone=dict(
        type='ResNetV1c',  # The type of backbone. Please refer to mmseg/backbone/resnet.py for details.
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
        num_classes=19,  # Number of segmentation classs. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
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
        num_classes=19,  # Number of segmentation classs. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # The configuration of norm layer.
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Config of loss function for the decode_head.
            type='CrossEntropyLoss',  # Type of loss used for segmentation.
            use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
            loss_weight=0.4)))  # Loss weight of auxiliary head, which is usually 0.4 of decode head.
train_cfg = dict()  # train_cfg is just a place holder for now.
test_cfg = dict(mode='whole')  # The test mode, options are 'whole' and 'sliding'. 'whole': whole image fully-convolutional test. 'sliding': sliding crop window on the image.
dataset_type = 'CityscapesDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/cityscapes/'  # Root path of data.
img_norm_cfg = dict(  # Image normalization config to normalize the input images.
    mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models.
    std=[58.395, 57.12, 57.375],  # Standard variance used to pre-training the pre-trained backbone models.
    to_rgb=True)  # The channel orders of image used to pre-training the pre-trained backbone models.
crop_size = (512, 1024)  # The crop size during training.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
    dict(type='Resize',  # Augmentation pipeline that resize the images and their annotations.
        img_scale=(2048, 1024),  # The largest scale of image.
        ratio_range=(0.5, 2.0)), # The augmented scale range as ratio.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=(512, 1024),  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(
        type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        flip_ratio=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    dict(
        type='Normalize',  # Augmentation pipeline that normalize the input images
        mean=[123.675, 116.28, 103.53],  # These keys are the same of img_norm_cfg since the
        std=[58.395, 57.12, 57.375],  # keys of img_norm_cfg are used here as arguments
        to_rgb=True),
    dict(type='Pad',  # Augmentation pipeline that pad the image to specified size.
        size=(512, 1024),  # The output size of padding.
        pad_val=0,  # The padding value for image.
        seg_pad_val=255),  # The padding value of 'gt_semantic_seg'.
    dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the segmentor
        keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the test time augmentations
        img_scale=(2048, 1024),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                 keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be supressed by the img_scale set above.
            dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used when flip=False
            dict(
                type='Normalize',  # Normalization config, the values are from img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', # Convert image to tensor
                keys=['img']),
            dict(type='Collect', # Collect pipeline that collect necessary keys for testing.
                keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
        type='CityscapesDataset',  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root='data/cityscapes/',  # The root of dataset.
        img_dir='leftImg8bit/train',  # The image directory of dataset.
        ann_dir='gtFine/train',  # The annotation directory of dataset.
        pipeline=[  # pipeline, this is passed by the train_pipeline created before.
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(  # Validation dataset config
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
cudnn_benchmark = True  # Whether use cudnn_benchmark to speed up, which is fast for fixed input size.
optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',  # Type of optimizers, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
    lr=0.01,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
    momentum=0.9,  # Momentum
    weight_decay=0.0005)  # Weight decay of SGD
optimizer_config = dict()  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
lr_config = dict(
    policy='poly',  # The policy of scheduler, also support Step, CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    power=0.9,  # The power of polynomial decay.
    min_lr=0.0001,  # The minimum learning rate to stable the training.
    by_epoch=False)  # Whether count by epoch or not.
runner = dict(
    type='IterBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_iters=40000) # Total number of iterations. For EpochBasedRunner use `max_epochs`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=4000)  # The save interval.
evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaulation/eval_hook.py for details.
    interval=4000,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.


```

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) for simple inllustration.

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
For example, we would like to change multi scale strategy to train/test a PSPNet. `train_pipeline`/`test_pipeline` are intermediate variable we would like modify.

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
