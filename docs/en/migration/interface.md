# Migration from MMSegmentation 0.x

## Introduction

This guide describes the fundamental differences between MMSegmentation 0.x and MMSegmentation 1.x in terms of behaviors and the APIs, and how these all relate to your migration journey.

## New dependencies

MMSegmentation 1.x depends on some new packages, you can prepare a new clean environment and install again according to the [installation tutorial](get_started.md).
Or install the below packages manually.

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture, and we splited many compentents unrelated to computer vision from MMCV to MMEngine.

2. [MMCV](https://github.com/open-mmlab/mmcv): The computer vision package of OpenMMLab. This is not a new dependency, but you need to upgrade it to above **2.0.0rc1** version.

3. [MMClassification](https://github.com/open-mmlab/mmclassification)(Optional): The  image classification toolbox and benchmark of OpenMMLab. This is not a new dependency, but you need to upgrade it to above **1.0.0rc0** version.

## Train launch

The main improvement of OpenMMLab 2.0 is releasing MMEngine which provides universal and powerful runner for unified interfaces to launch training jobs.

Compared with MMSeg0.x, MMSeg1.x provides fewer command line arguments in `tools/train.py`

<table class="docutils">
<tr>
<td>Function</td>
<td>Original</td>
<td>New</td>
</tr>
<tr>
<td>Loading pre-trained checkpoint</td>
<td>--load_from=$CHECKPOINT</td>
<td>--cfg-options load_from=$CHECKPOINT</td>
</tr>
<tr>
<td>Resuming Train from specific checkpoint</td>
<td>--resume-from=$CHECKPOINT</td>
<td>--resume=$CHECKPOINT</td>
</tr>
<tr>
<td>Resuming Train from the latest checkpoint</td>
<td>--auto-resume</td>
<td>--resume='auto'</td>
</tr>
<tr>
<td>Whether not to evaluate the checkpoint during training</td>
<td>--no-validate</td>
<td>--cfg-options val_cfg=None val_dataloader=None val_evaluator=None</td>
</tr>
<tr>
<td>Training device assignment</td>
<td>--gpu-id=$DEVICE_ID</td>
<td>-</td>
</tr>
<tr>
<td>Whether or not set different seeds for different ranks</td>
<td>--diff-seed</td>
<td>--cfg-options randomness.diff_rank_seed=True</td>
</tr>
<td>Whether to set deterministic options for CUDNN backend</td>
<td>--deterministic</td>
<td>--cfg-options randomness.deterministic=True</td>
</table>

## Configuration file

### Model settings

No changes in `model.backbone`, `model.neck`, `model.decode_head` and `model.losses` fields.

Add `model.data_preprocessor` field to configure the `DataPreProcessor`, including:

- `mean`(Sequence, optional): The pixel mean of R, G, B channels. Defaults to None.

- `std`(Sequence, optional): The pixel standard deviation of R, G, B channels. Defaults to None.

- `size`(Sequence, optional): Fixed padding size.

- `size_divisor` (int, optional): The divisor of padded size.

- `seg_pad_val` (float, optional): Padding value of segmentation map. Default: 255.

- `padding_mode` (str): Type of padding. Default: 'constant'.

  - constant: pads with a constant value, this value is specified with pad_val.

- `bgr_to_rgb` (bool): whether to convert image from BGR to RGB.Defaults to False.

- `rgb_to_bgr` (bool): whether to convert image from RGB to RGB. Defaults to False.

**Note:**
Please refer [models documentation](../advanced_guides/models.md) for more details.

### Dataset settings

Changes in **data**:

The original `data` field is split to `train_dataloader`, `val_dataloader` and `test_dataloader`. This allows us to configure them in fine-grained. For example, you can specify different sampler and batch size during training and test.
The `samples_per_gpu` is renamed to `batch_size`.
The `workers_per_gpu` is renamed to `num_workers`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(...),
    val=dict(...),
    test=dict(...),
)
```

</td>
<tr>
<td>New</td>
<td>

```python
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # necessary
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

Changes in **pipeline**

- The original formatting transforms **`ToTensor`**、**`ImageToTensor`**、**`Collect`** are combined as [`PackSegInputs`](mmseg.datasets.transforms.PackSegInputs)
- We don't recommend to do **`Normalize`** and **Pad** in the dataset pipeline. Please remove it from pipelines and set it in the `data_preprocessor` field.
- The original **`Resize`** in MMSeg 1.x has been changed to **`RandomResize`** and the input arguments `img_scale` is renamed to `scale`, and the default value of `keep_ratio` is modified to False.

**Note:**
We move some work of data transforms to the data preprocessor, like normalization, see [the documentation](package.md) for more details.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
```

</td>
<tr>
<td>New</td>
<td>

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2560, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
```

</td>
</tr>
</table>

Changes in **`evaluation`**:

- The **`evaluation`** field is split to `val_evaluator` and `test_evaluator`. And it won't support `interval` and `save_best` arguments.
  The `interval` is moved to `train_cfg.val_interval`, and the `save_best`
  is moved to `default_hooks.checkpoint.save_best`. `pre_eval` has been removed.
- `'mIoU'` has been changed to `'IoUMetric'`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
```

</td>
<tr>
<td>New</td>
<td>

```python
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

</td>
</tr>
</table>

### Optimizer and Schedule settings

Changes in **`optimizer`** and **`optimizer_config`**:

- Now we use `optim_wrapper` field to specify all configuration about the optimization process. And the
  `optimizer` is a sub field of `optim_wrapper` now.
- `paramwise_cfg` is also a sub field of `optim_wrapper`, instead of `optimizer`.
- `optimizer_config` is removed now, and all configurations of it are moved to `optim_wrapper`.
- `grad_clip` is renamed to `clip_grad`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
```

</td>
<tr>
<td>New</td>
<td>

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0005),
    clip_grad=dict(max_norm=1, norm_type=2))
```

</td>
</tr>
</table>

Changes in **`lr_config`**:

- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `warmup` related arguments are removed, since we use schedulers combination to implement this
  functionality.

The new schedulers combination mechanism is very flexible, and you can use it to design many kinds of learning
rate / momentum curves. See [the tutorial](TODO) for more details.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
```

</td>
<tr>
<td>New</td>
<td>

```python
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
```

</td>
</tr>
</table>

Changes in **`runner`**:

Most configuration in the original `runner` field is moved to `train_cfg`, `val_cfg` and `test_cfg`, which
configure the loop in training, validation and test.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
runner = dict(type='IterBasedRunner', max_iters=20000)
```

</td>
<tr>
<td>New</td>
<td>

```python
# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop') # Use the default validation loop.
test_cfg = dict(type='TestLoop') # Use the default test loop.
```

</td>
</tr>
</table>

In fact, in OpenMMLab 2.0, we introduced `Loop` to control the behaviors in training, validation and test. The functionalities of `Runner` are also changed. You can find more details of [runner tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/runner.md)
in [MMEngine](https://github.com/open-mmlab/mmengine/).

### Runtime settings

Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.
And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

```python
default_hooks = dict(
    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint every 2000 iterations.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),

    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization.
    visualization=dict(type='SegVisualizationHook'))
```

In addition, we split the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal and TensorBoard.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
```

</td>
<tr>
<td>New</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

</td>
</tr>
</table>

Changes in **`load_from`** and **`resume_from`**:

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.
  - If `resume=True` and `load_from` is **not None**, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is **None**, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is **not None**, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is **None**, do not load nor resume.

Changes in **`dist_params`**: The `dist_params` field is a sub field of `env_cfg` now. And there are some new
configurations in the `env_cfg`.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```

Changes in **`workflow`**: `workflow` related functionalities are removed.

New field **`visualizer`**: The visualizer is a new design in OpenMMLab 2.0 architecture. We use a
visualizer instance in the runner to handle results & log visualization and save to different backends.
See the [visualization tutorial](user_guides/visualization.md) for more details.

New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMSegmentation is `mmseg`. See [the registry tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md) for more details.
