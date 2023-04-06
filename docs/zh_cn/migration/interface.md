# 从 MMSegmentation 0.x 迁移

## 引言

本指南介绍了 MMSegmentation 0.x 和 MMSegmentation1.x 在表现和 API 方面的基本区别，以及这些与迁移过程的关系。

## 新的依赖

MMSegmentation 1.x 依赖于一些新的软件包，您可以准备一个新的干净环境，然后根据[安装教程](../get_started.md)重新安装。

或手动安装以下软件包。

1. [MMEngine](https://github.com/open-mmlab/mmengine)：MMEngine 是 OpenMMLab 2.0 架构的核心，我们将许多与计算机视觉无关的内容从 MMCV 拆分到 MMEngine 中。

2. [MMCV](https://github.com/open-mmlab/mmcv)：OpenMMLab 的计算机视觉包。这不是一个新的依赖，但您需要将其升级到 **2.0.0** 或以上的版本。

3. [MMClassification](https://github.com/open-mmlab/mmclassification)（可选）：OpenMMLab 的图像分类工具箱和基准。这不是一个新的依赖，但您需要将其升级到 **1.0.0rc6** 版本。

4. [MMDetection](https://github.com/open-mmlab/mmdetection)(可选): OpenMMLab 的目标检测工具箱和基准。这不是一个新的依赖，但您需要将其升级到 **3.0.0** 或以上的版本。

## 启动训练

OpenMMLab 2.0 的主要改进是发布了 MMEngine，它为启动训练任务的统一接口提供了通用且强大的执行器。

与 MMSeg 0.x 相比，MMSeg 1.x 在 `tools/train.py` 中提供的命令行参数更少

<table class="docutils">
<tr>
<td>功能</td>
<td>原版</td>
<td>新版</td>
</tr>
<tr>
<td>加载预训练模型</td>
<td>--load_from=$CHECKPOINT</td>
<td>--cfg-options load_from=$CHECKPOINT</td>
</tr>
<tr>
<td>从特定检查点恢复训练</td>
<td>--resume-from=$CHECKPOINT</td>
<td>--resume=$CHECKPOINT</td>
</tr>
<tr>
<td>从最新的检查点恢复训练</td>
<td>--auto-resume</td>
<td>--resume='auto'</td>
</tr>
<tr>
<td>训练期间是否不评估检查点</td>
<td>--no-validate</td>
<td>--cfg-options val_cfg=None val_dataloader=None val_evaluator=None</td>
</tr>
<tr>
<td>指定训练设备</td>
<td>--gpu-id=$DEVICE_ID</td>
<td>-</td>
</tr>
<tr>
<td>是否为不同进程设置不同的种子</td>
<td>--diff-seed</td>
<td>--cfg-options randomness.diff_rank_seed=True</td>
</tr>
<td>是否为 CUDNN 后端设置确定性选项</td>
<td>--deterministic</td>
<td>--cfg-options randomness.deterministic=True</td>
</table>

## 测试启动

与训练启动类似，MMSegmentation 1.x 的测试启动脚本在 tools/test.py 中仅提供关键命令行参数，以下是测试启动脚本的区别，更多关于测试启动的细节请参考[这里](../user_guides/4_train_test.md)。

<table class="docutils">
<tr>
<td>功能</td>
<td>0.x</td>
<td>1.x</td>
</tr>
<tr>
<td>指定评测指标</td>
<td>--eval mIoU</td>
<td>--cfg-options test_evaluator.type=IoUMetric</td>
</tr>
<tr>
<td>测试时数据增强</td>
<td>--aug-test</td>
<td>--tta</td>
</tr>
<tr>
<td>测试时是否只保存预测结果不计算评测指标</td>
<td>--format-only</td>
<td>--cfg-options test_evaluator.format_only=True</td>
</tr>
</table>

## 配置文件

### 模型设置

`model.backend`、`model.neck`、`model.decode_head` 和 `model.loss` 字段没有更改。

添加 `model.data_preprocessor` 字段以配置 `DataPreProcessor`，包括：

- `mean`（Sequence，可选）：R、G、B 通道的像素平均值。默认为 None。

- `std`（Sequence，可选）：R、G、B 通道的像素标准差。默认为 None。

- `size`（Sequence，可选）：固定的填充大小。

- `size_divisor`（int，可选）：填充图像可以被当前值整除。

- `seg_pad_val`（float，可选）：分割图的填充值。默认值：255。

- `padding_mode`（str）：填充类型。默认值：'constant'。

  - constant：常量值填充，值由 pad_val 指定。

- `bgr_to_rgb`（bool）：是否将图像从 BGR 转换为 RGB。默认为 False。

- `rgb_to_bgr`（bool）：是否将图像从 RGB 转换为 BGR。默认为 False。

**注：**
有关详细信息，请参阅[模型文档](../advanced_guides/models.md)。

### 数据集设置

**data** 的更改：

原版 `data` 字段被拆分为 `train_dataloader`、`val_dataloader` 和 `test_dataloader`，允许我们以细粒度配置它们。例如，您可以在训练和测试期间指定不同的采样器和批次大小。
`samples_per_gpu` 重命名为 `batch_size`。
`workers_per_gpu` 重命名为 `num_workers`。

<table class="docutils">
<tr>
<td>原版</td>
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
<td>新版</td>
<td>

```python
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # 必须
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # 必须
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

**数据增强变换流程**变更

- 原始格式转换 **`ToTensor`**、**`ImageToTensor`**、**`Collect`** 组合为 [`PackSegInputs`](mmseg.datasets.transforms.PackSegInputs)
- 我们不建议在数据集流程中执行 **`Normalize`** 和 **Pad**。请将其从流程中删除，并将其设置在 `data_preprocessor` 字段中。
- MMSeg 1.x 中原始的 **`Resize`** 已更改为 **`RandomResize `**，输入参数 `img_scale` 重命名为 `scale`，`keep_ratio` 的默认值修改为 False。
- 原始的 `test_pipeline` 将单尺度和多尺度测试结合在一起，在 MMSeg 1.x 中，我们将其分为 `test_pipeline` 和 `tta_pipeline`。

**注：**
我们将一些数据转换工作转移到数据预处理器中，如归一化，请参阅[文档](package.md)了解更多详细信息。

训练流程

<table class="docutils">
<tr>
<td>原版</td>
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
<td>新版</td>
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

测试流程

<table class="docutils">
<tr>
<td>原版</td>
<td>

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

</td>
<tr>
<td>新版</td>
<td>

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
```

</td>
</tr>
</table>

**`evaluation`** 中的更改：

- **`evaluation`** 字段被拆分为 `val_evaluator` 和 `test_evaluator `。而且不再支持 `interval` 和 `save_best` 参数。
  `interval` 已移动到 `train_cfg.val_interval`，`save_best` 已移动到 `default_hooks.checkpoint.save_best`。`pre_eval` 已删除。
- `IoU` 已更改为 `IoUMetric`。

<table class="docutils">
<tr>
<td>原版</td>
<td>

```python
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
```

</td>
<tr>
<td>新版</td>
<td>

```python
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

</td>
</tr>
</table>

### Optimizer 和 Schedule 设置

**`optimizer`** 和 **`optimizer_config`** 中的更改：

- 现在我们使用 `optim_wrapper` 字段来指定优化过程的所有配置。以及 `optimizer` 是 `optim_wrapper` 的一个子字段。
- `paramwise_cfg` 也是 `optim_wrapper` 的一个子字段，以替代 `optimizer`。
- `optimizer_config` 现在被删除，它的所有配置都被移动到 `optim_wrapper` 中。
- `grad_clip` 重命名为 `clip_grad`。

<table class="docutils">
<tr>
<td>原版</td>
<td>

```python
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
```

</td>
<tr>
<td>新版</td>
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

**`lr_config`** 中的更改：

- 我们将 `lr_config` 字段删除，并使用新的 `param_scheduler` 替代。
- 我们删除了与 `warmup` 相关的参数，因为我们使用 scheduler 组合来实现该功能。

新的 scheduler 组合机制非常灵活，您可以使用它来设计多种学习率/动量曲线。有关详细信息，请参见[教程](TODO)。

<table class="docutils">
<tr>
<td>原版</td>
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
<td>新版</td>
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

**`runner`** 中的更改：

原版 `runner` 字段中的大多数配置被移动到 `train_cfg`、`val_cfg` 和 `test_cfg` 中，以在训练、验证和测试中配置 loop。

<table class="docutils">
<tr>
<td>原版</td>
<td>

```python
runner = dict(type='IterBasedRunner', max_iters=20000)
```

</td>
<tr>
<td>新版</td>
<td>

```python
# `val_interval` 是旧版本的 `evaluation.interval`。
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop') # 使用默认的验证循环。
test_cfg = dict(type='TestLoop') # 使用默认的测试循环。
```

</td>
</tr>
</table>

事实上，在 OpenMMLab 2.0 中，我们引入了 `Loop` 来控制训练、验证和测试中的行为。`Runner` 的功能也发生了变化。您可以在 [MMMEngine](https://github.com/open-mmlab/mmengine/) 的[执行器教程](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/runner.md) 中找到更多的详细信息。

### 运行时设置

**`checkpoint_config`** 和 **`log_config`** 中的更改：

`checkpoint_config` 被移动到 `default_hooks.checkpoint` 中，`log_config` 被移动到 `default_hooks.logger` 中。
并且我们将许多钩子设置从脚本代码移动到运行时配置的 `default_hooks` 字段中。

```python
default_hooks = dict(
    # 记录每次迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每50次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),

    # 启用参数调度程序。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每2000次迭代保存一次检查点。
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 验证结果可视化。
    visualization=dict(type='SegVisualizationHook'))
```

此外，我们将原版 logger 拆分为 logger 和 visualizer。logger 用于记录信息，visualizer 用于在不同的后端显示 logger，如 terminal 和 TensorBoard。

<table class="docutils">
<tr>
<td>原版</td>
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
<td>新版</td>
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

**`load_from`** 和 **`resume_from`** 中的更改：

- 删除 `resume_from`。我们使用 `resume` 和 `load_from` 来替换它。
  - 如果 `resume=True` 且 `load_from` 为 **not None**，则从 `load_from` 中的检查点恢复训练。
  - 如果 `resume=True` 且 `load_from` 为 **None**，则尝试从工作目录中的最新检查点恢复。
  - 如果 `resume=False` 且 `load_from` 为 **not None**，则只加载检查点，而不继续训练。
  - 如果 `resume=False` 且 `load_from` 为 **None**，则不加载或恢复。

**`dist_params`** 中的更改：`dist_params` 字段现在是 `env_cfg` 的子字段。并且 `env_cfg` 中还有一些新的配置。

```python
env_cfg = dict(
    # 是否启用 cudnn_benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)
```

**`workflow`** 的改动:`workflow` 相关功能被删除。

新字段 **`visualizer`**：visualizer 是 OpenMMLab 2.0 体系结构中的新设计。我们在 runner 中使用 visualizer 实例来处理结果和日志可视化，并保存到不同的后端。更多详细信息，请参阅[可视化教程](../user_guides/visualization.md)。

新字段 **`default_scope`**：搜索所有注册模块的起点。MMSegmentation 中的 `default_scope` 为 `mmseg`。请参见[注册器教程](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/registry.md)了解更多详情。
