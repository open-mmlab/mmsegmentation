# 训练引擎

MMEngine 定义了一些[基础循环控制器](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) 例如基于轮次的训练循环 (`EpochBasedTrainLoop`), 基于迭代次数的训练循环 (`IterBasedTrainLoop`), 标准的验证循环 (`ValLoop`) 和标准的测试循环 (`TestLoop`).
OpenMMLab 的算法库如 MMSegmentation 将模型训练, 测试和推理抽象为执行器(`Runner`) 来处理. 用户可以直接使用 MMEngine 中的默认执行器, 也可以对执行器进行修改以满足定制化需求. 这个文档主要介绍用户如何配置已有的运行设定, 钩子和优化器的基本概念与使用方法.

## 配置运行设定

### 配置训练长度

循环控制器指的是训练, 验证和测试时的执行流程, 在配置文件里面使用 `train_cfg`, `val_cfg` 和 `test_cfg` 来构建这些流程. MMSegmentation 在 `configs/_base_/schedules` 文件夹里面的 `train_cfg` 设置常用的训练长度.
例如, 使用基于迭代次数的训练循环 (`IterBasedTrainLoop`) 去训练 80,000 个迭代次数, 并且每 8,000 iteration 做一次验证, 可以如下设置:

```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
```

### 配置训练优化器

这里是一个 SGD 优化器的例子:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
```

OpenMMLab 支持 PyTorch 里面所有的优化器, 更多细节可以参考 MMEngine [优化器文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

需要强调的是, `optim_wrapper` 是 `runner` 的变量, 所以需要配置优化器时配置的字段是 `optim_wrapper` 字段. 更多关于优化器的使用方法, 可以看下面优化器的章节.

### 配置训练参数调度器

在配置训练参数调度器前, 推荐先了解 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md) 里面关于参数调度器的基本概念.

以下是一个参数调度器的例子, 训练时前 1,000 个 iteration 时采用线性变化的学习率策略作为训练预热, 从 1,000 iteration 之后直到最后 16,000 个 iteration 时则采用默认的多项式学习率衰减:

```python
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
```

注意: 当修改 `train_cfg` 里面 `max_iters` 的时候, 请确保参数调度器 `param_scheduler` 里面的参数也被同时修改.

## 钩子 (Hook)

### 介绍

OpenMMLab 将模型训练和测试过程抽象为 `Runner`, 插入钩子可以实现在 `Runner` 中不同的训练和测试节点 (例如 "每个训练 iter 前后", "每个验证 iter 前后" 等不同阶段) 所需要的相应功能. 更多钩子机制的介绍可以参考[这里](https://www.calltutors.com/blog/what-is-hook).

`Runner` 中所使用的钩子分为两类:

- 默认钩子 (default hooks)

它们实现了训练时所必需的功能, 在配置文件中用 `default_hooks` 定义传给 `Runner`, `Runner` 通过 [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/090104df21acd05a8aadae5a0d743a7da3314f6f/mmengine/runner/runner.py#L1780) 方法注册.
钩子有对应的优先级, 优先级越高, 越早被执行器调用. 如果优先级一样, 被调用的顺序和钩子注册的顺序一致.
不建议用户修改默认钩子的优先级, 可以参考 [mmengine hooks 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md) 了解钩子优先级的定义.
下面是 MMSegmentation 中所用到的默认钩子：

|                                                           钩子                                                            |                                              功能                                               |      优先级       |
| :-----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: | :---------------: |
|            [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)            |                                   记录 iteration 花费的时间.                                    |    NORMAL (50)    |
|               [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)                | 从 `Runner` 里不同的组件中收集日志记录, 并将其输出到终端, JSON 文件, tensorboard, wandb 等下游. | BELOW_NORMAL (60) |
|       [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py)       |                          更新优化器里面的一些超参数, 例如学习率的动量.                          |     LOW (70)      |
|           [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)            |                                  规律性地保存 checkpoint 文件.                                  |   VERY_LOW (90)   |
|        [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)        |                               确保分布式采样器 shuffle 是打开的.                                |    NORMAL (50)    |
| [SegVisualizationHook](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/visualization/local_visualizer.py) |                                可视化验证和测试过程里的预测结果.                                |    NORMAL (50)    |

MMSegmentation 会在 [`defualt_hooks`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/schedules/schedule_160k.py#L19-L25) 里面注册一些训练所必需功能的钩子::

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=32000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

以上默认钩子除 `SegVisualizationHook` 外都是在 MMEngine 中所实现, `SegVisualizationHook` 是在 MMSegmentation 里被实现的钩子, 之后会专门介绍.

- 修改默认的钩子

以 `default_hooks` 里面的 `logger` 和 `checkpoint` 为例, 我们来介绍如何修改 `default_hooks` 中默认的钩子.

(1) 模型保存配置
`default_hooks` 使用 `checkpoint` 字段来初始化[模型保存钩子 (CheckpointHook)](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19).

```python
checkpoint = dict(type='CheckpointHook', interval=1)
```

用户可以设置 `max_keep_ckpts` 来只保存少量的检查点或者用 `save_optimizer` 来决定是否保存 optimizer 的信息.
更多相关参数的细节可以参考[这里](https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.hooks.CheckpointHook.html#checkpointhook).

(2) 日志配置
`日志钩子 (LoggerHook)` 被用来收集 `执行器 (Runner)` 里面不同组件的日志信息然后写入终端, JSON 文件, tensorboard 和 wandb 等地方.

```python
logger=dict(type='LoggerHook', interval=10)
```

在最新的 1.x 版本的 MMSegmentation 里面, 一些日志钩子 (LoggerHook) 例如 `TextLoggerHook`, `WandbLoggerHook` 和 `TensorboardLoggerHook` 将不再被使用.
作为替代, MMEngine 使用 `LogProcessor` 来处理上述钩子处理的信息, 它们现在在 [`MessageHub`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/logging/message_hub.py#L17),
[`WandbVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L324) 和 [`TensorboardVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L472) 里面.

具体使用方法如下, 配置可视化器和同时指定可视化后端, 这里使用 Tensorboard 作为可视化器的后端:

```python
# TensorboardVisBackend
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=[dict(type='TensorboardVisBackend')], name='visualizer')
```

关于更多相关用法, 可以参考 [MMEngine 可视化后端用户教程](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md).

- 自定义钩子 (custom hooks)

自定义钩子在配置通过 `custom_hooks` 定义, `Runner` 通过 [`register_custom_hooks`](https://github.com/open-mmlab/mmengine/blob/090104df21acd05a8aadae5a0d743a7da3314f6f/mmengine/runner/runner.py#L1852) 方法注册.
自定义钩子优先级需要在配置文件里设置, 如果没有设置, 则会被默认设置为 `NORMAL`. 下面是部分 MMEngine 中实现的自定义钩子:

|                                                  钩子                                                  |                                             用法                                             |
| :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
|         [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)         |               在模型训练时使用指数滑动平均 (Exponential Moving Average, EMA).                |
| [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)  |                          在训练时释放所有没有被缓存占用的 GPU 显存.                          |
| [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py) | 在每个训练 Epoch 结束时同步模型 buffer 里的参数例如 BN 里的 `running_mean` 和 `running_var`. |

以下是 `EMAHook` 的用例, 配置文件中, 将已经实现的自定义钩子的配置作为 `custom_hooks` 列表中的成员.

```python
custom_hooks = [
    dict(type='EMAHook', start_iters=500, priority='NORMAL')
]
```

### SegVisualizationHook

MMSegmentation 实现了 [`SegVisualizationHook`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/engine/hooks/visualization_hook.py#L17), 用来在验证和测试时可视化预测结果.
`SegVisualizationHook` 重写了基类 `Hook` 中的 `_after_iter` 方法, 在验证或测试时, 根据指定的迭代次数间隔调用 `visualizer` 的 `add_datasample` 方法绘制语义分割结果, 具体实现如下:

```python
...
@HOOKS.register_module()
class SegVisualizationHook(Hook):
...
    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
...
        # 如果是训练阶段或者 self.draw 为 False 则直接跳出
        if self.draw is False or mode == 'train':
            return
...
        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)

```

关于可视化更多的细节可以查看[这里](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/visualization.md).

## 优化器

在上面配置运行设定里, 我们给出了配置训练优化器的简单示例. 本章节将进一步详细介绍在 MMSegmentation 里如何配置优化器.

### 优化器封装

OpenMMLab 2.0 设计了优化器封装, 它支持不同的训练策略, 包括混合精度训练、梯度累加和梯度截断等, 用户可以根据需求选择合适的训练策略.
优化器封装还定义了一套标准的参数更新流程, 用户可以基于这一套流程, 在同一套代码里, 实现不同训练策略的切换. 如果想了解更多, 可以参考 [MMEngine 优化器封装文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

以下是 MMSegmentation 中常用的使用方法:

#### 配置 PyTorch 支持的优化器

OpenMMLab 2.0 支持 PyTorch 原生所有优化器, 参考[这里](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md#%E7%AE%80%E5%8D%95%E9%85%8D%E7%BD%AE).

在配置文件中设置训练时 `Runner` 所使用的优化器, 需要定义 `optim_wrapper`, 而不是 `optimizer`, 下面是一个配置训练中优化器的例子:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
```

#### 配置梯度裁剪

当模型训练需要使用梯度裁剪的训练技巧式, 可以按照如下示例进行配置:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer,
                        clip_grad=dict(max_norm=0.01, norm_type=2))
```

这里 `max_norm` 指的是裁剪后梯度的最大值,  `norm_type` 指的是裁剪梯度时使用的范数. 相关方法可参考 [torch.nn.utils.clip_grad_norm\_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

#### 配置混合精度训练

当需要使用混合精度训练降低内存时, 可以使用 `AmpOptimWrapper`, 具体配置如下:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

[`AmpOptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/amp_optimizer_wrapper.py#L20) 中 `loss_scale` 的默认设置是 `dynamic`.

#### 配置模型网络不同层的超参数

在模型训练中, 如果想在优化器里为不同参数分别设置优化策略, 例如设置不同的学习率、权重衰减等超参数, 可以通过设置配置文件里 `optim_wrapper` 中的 `paramwise_cfg` 来实现.

下面的配置文件以 [ViT `optim_wrapper`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py#L15-L27) 为例介绍 `paramwise_cfg` 参数使用.
训练时将 `pos_embed`, `mask_token`, `norm` 模块的 weight decay 参数的系数设置成 0.
即: 在训练时, 这些模块的 weight decay 将被变为 `weight_decay * decay_mult`=0.

```python
optimizer = dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
```

其中 `decay_mult` 指的是对应参数的权重衰减的系数.
关于更多 `paramwise_cfg` 的使用可以在 [MMEngine 优化器封装文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md) 里面查到.

### 优化器封装构造器

默认的优化器封装构造器 [`DefaultOptimWrapperConstructor`](https://github.com/open-mmlab/mmengine/blob/376251961da47ea8254ab808ae5c51e1430f18dc/mmengine/optim/optimizer/default_constructor.py#L19) 根据输入的 `optim_wrapper` 和 `optim_wrapper` 中定义的 `paramwise_cfg` 来构建训练中使用的优化器. 当 [`DefaultOptimWrapperConstructor`](https://github.com/open-mmlab/mmengine/blob/376251961da47ea8254ab808ae5c51e1430f18dc/mmengine/optim/optimizer/default_constructor.py#L19) 功能不能满足需求时, 可以自定义优化器封装构造器来实现超参数的配置.

MMSegmentation 中的实现了 [`LearningRateDecayOptimizerConstructor`](https://github.com/open-mmlab/mmsegmentation/blob/b21df463d47447f33c28d9a4f46136ad64d34a40/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py#L104), 可以对以 ConvNeXt, BEiT 和 MAE 为骨干网络的模型训练时, 骨干网络的模型参数的学习率按照定义的衰减比例（`decay_rate`）逐层递减, 在配置文件中的配置如下:

```python
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')
```

`_delete_=True` 的作用是 OpenMMLab Config 中的忽略继承的配置, 在该代码片段中忽略继承的 `optim_wrapper` 配置, 更多 `_delete_` 字段的内容可以参考 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/config.md#%E5%88%A0%E9%99%A4%E5%AD%97%E5%85%B8%E4%B8%AD%E7%9A%84-key).
