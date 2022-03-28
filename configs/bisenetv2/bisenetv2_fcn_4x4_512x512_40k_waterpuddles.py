### Only need to edit this config file to supercede the settings in other config files.s
# import os
### Base config files used within configs/_base_
_base_ = [
    '../_base_/models/bisenetv2.py',    # model settings
    '../_base_/datasets/waterpuddles.py', # dataset settings
    '../_base_/schedules/schedule_40k.py', # scheduler settings
    '../_base_/default_runtime.py' # other runtime settings
]
# base_workdir = "/home/mind02/mmsegmentation/train_runs"
# current_workdir = "waterpuddles"
# new_workdir = os.path.join(base_workdir,current_workdir)
# if os.path.exists(new_workdir):
    # for i in range(100):
        # temp_workdir = os.path.join(new_workdir, str(i))
        # if os.path.exists(temp_workdir):
            # continue
        # else:
            # work_dir = temp_workdir
work_dir = "train_runs/puddle_1000_chasedb"
### From configs/_base_/models configs
# model settings
# norm_cfg = dict(type='BN', requires_grad=True)  # Segmentation usually uses SyncBN for multiple GPU training, BN if using 1 GPU
# model = dict(
#     type='EncoderDecoder',  # Name of segmentor
#     pretrained=None,  # The ImageNet pretrained backbone to be loaded
#     backbone=dict(
#         type='BiSeNetV2',
#         detail_channels=(64, 64, 128),
#         semantic_channels=(16, 32, 64, 128),
#         semantic_expansion_ratio=6,
#         bga_channels=128,
#         out_indices=(0, 1, 2, 3, 4),
#         init_cfg=None,
#         align_corners=False),
#     decode_head=dict(
#         type='FCNHead',  # Type of decode head. Please refer to mmseg/models/decode_heads for available options.
#         in_channels=128,  # Input channel of decode head.
#         in_index=0,  # The index of feature map to select.
#         channels=1024,  # The intermediate channels of decode head.
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,  # The dropout ratio before final classification layer.
#         num_classes=1,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
#         norm_cfg=norm_cfg,  # The configuration of norm layer.
#         align_corners=False,  # The align_corners argument for resize in decoding.
#         loss_decode=dict(  # Config of loss function for the decode_head.
#             type='CrossEntropyLoss',  # Type of loss used for segmentation.
#             use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
#             loss_weight=1.0)),  # Loss weight of decode head.
#     auxiliary_head=[dict(
#         type='FCNHead',  # Type of auxiliary head. Please refer to mmseg/models/decode_heads for available options.
#         in_channels=16,  # Input channel of auxiliary head.
#         channels=16,  # The intermediate channels of decode head.
#         num_convs=2,  # Number of convs in FCNHead. It is usually 1 in auxiliary head.
#         num_classes=1,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
#         in_index=1,  # The index of feature map to select.
#         norm_cfg=norm_cfg,  # The configuration of norm layer.        
#         concat_input=False,  # Whether concat output of convs with input before classification layer.
#         align_corners=False,  # The align_corners argument for resize in decoding.        
#         loss_decode=dict(  # Config of loss function for the decode_head.
#             type='CrossEntropyLoss',  # Type of loss used for segmentation.
#             use_sigmoid=False,  # Whether use sigmoid activation for segmentation.
#             loss_weight=1.0)),  # Loss weight of auxiliary head, which is usually 0.4 of decode head.
#         dict(
#             type='FCNHead',
#             in_channels=32,
#             channels=64,
#             num_convs=2,
#             num_classes=19,
#             in_index=2,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             align_corners=False,
#             loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#         dict(
#             type='FCNHead',
#             in_channels=64,
#             channels=256,
#             num_convs=2,
#             num_classes=19,
#             in_index=3,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             align_corners=False,
#             loss_decode=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#         dict(
#             type='FCNHead',
#             in_channels=128,
#             channels=1024,
#             num_convs=2,
#             num_classes=19,
#             in_index=4,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             align_corners=False,
#             loss_decode=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
#         ]
#     )
# # model training and testing settings
# train_cfg=dict()
# test_cfg = dict(mode='whole')  # The test mode, options are 'whole' and 'sliding'. 'whole': whole image fully-convolutional test. 'sliding': sliding crop window on the image.
# ###############################################################

# ### From configs/_base_/datasets configs
# dataset_type = 'waterpuddlesDataset'
# data_root = 'data/waterpuddles'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_scale = (960, 999)
# crop_size = (128, 128)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
#     train=dict(
#         type='RepeatDataset',
#         times=40000,
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             img_dir='images/training',
#             ann_dir='annotations/training',
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='images/validation',
#         ann_dir='annotations/validation',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='images/validation',
#         ann_dir='annotations/validation',
#         pipeline=test_pipeline))
# #################################################

# ### From From configs/_base_/schedules/schedule_{chosen epochs}.py
# # chosen epochs can be 20k, 40k, 80k, 160k, 320k

# # optimizer
# optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
#     type='SGD',  # Type of optimizers, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
#     lr=0.01,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
#     momentum=0.9,  # Momentum
#     weight_decay=0.0005)  # Weight decay of SGD
# optimizer_config = dict()  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.

# # learning rate policy
# lr_config = dict(
#     policy='poly',  # The policy of scheduler, also support Step, CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
#     power=0.9,  # The power of polynomial decay.
#     min_lr=0.0001,  # The minimum learning rate to stable the training.
#     by_epoch=False)  # Whether count by epoch or not.

# # runtime settings
# runner = dict(
    # type='IterBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    # max_iters=160000) # Total number of iterations. For EpochBasedRunner use `max_epochs`
# checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    # by_epoch=False,  # Whether count by epoch or not.
    # interval=4000)  # The save interval.
# evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaluation/eval_hook.py for details.
    # interval=16000,  # The interval of evaluation.
    # metric='mIoU',  # The evaluation metric.
    # pre_eval=True)
# #####################################################

# ### From configs/_base_/default_runtime.py
# # yapf:disable
# log_config = dict(  # config to register logger hook
#     interval=50,  # Interval to print the log
#     hooks=[
#         # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
#         dict(type='TextLoggerHook', by_epoch=False)
#     ])
# # yapf:enable
# dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
# log_level = 'INFO'  # The level of logging.
# load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
# resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved.
# workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
# cudnn_benchmark = True  # Whether use cudnn_benchmark to speed up, which is fast for fixed input size.
# ####################################################
