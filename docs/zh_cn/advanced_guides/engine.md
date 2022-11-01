# 训练引擎

## 钩子 (Hook)

### 介绍

OpenMMLab 将模型训练和测试过程抽象为 `Runner`, 插入钩子可以实现在 `Runner` 中不同的训练和测试节点 (例如 "每个训练 iter 前后", "每个验证 iter 前后" 等不同阶段) 所需要的相应功能. 更多钩子机制的介绍可以参考[这里](https://www.calltutors.com/blog/what-is-hook).

`Runner` 中所使用的钩子分为两类:

- 默认钩子 (default hooks)

它们实现了训练时所必需的功能，在配置文件中用 `default_hooks` 定义传给 `Runner`, `Runner` 通过 [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/090104df21acd05a8aadae5a0d743a7da3314f6f/mmengine/runner/runner.py#L1780) 方法注册.
钩子有对应的优先级, 优先级越高, 越早被执行器调用. 如果优先级一样, 被调用的顺序和钩子注册的顺序一致.
不建议用户修改默认钩子的优先级，可以参考 [mmengine hooks 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md) 了解钩子优先级的定义.
下面是 MMSegmentation 中所用到的默认钩子：

|                                                           钩子                                                            |                                               用法                                               |      优先级       |
| :-----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :---------------: |
|            [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)            |                                    记录 iteration 花费的时间.                                    |    NORMAL (50)    |
|               [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)                | 从 `Runner` 里不同的组件中收集日志记录，并将其输出到终端， JSON 文件，tensorboard，wandb 等下游. | BELOW_NORMAL (60) |
|       [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py)       |                          更新优化器里面的一些超参数，例如学习率的动量.                           |     LOW (70)      |
|           [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)            |                                  规律性地保存 checkpoint 文件.                                   |   VERY_LOW (90)   |
|        [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)        |                                确保分布式采样器 shuffle 是打开的.                                |    NORMAL (50)    |
| [SegVisualizationHook](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/visualization/local_visualizer.py) |                                可视化验证和测试过程里的预测结果.                                 |    NORMAL (50)    |

它们在配置文件中的配置为:

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
`SegVisualizationHook` 重写了基类 `Hook` 中的 `_after_iter` 方法, 在验证或测试时, 根据指定的迭代次数间隔调用 `visualizer` 的 `add_datasample` 方法绘制语义分割结果，具体实现如下:

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

### 优化器封装

OpenMMLab 2.0 设计了优化器封装, 它支持不同的训练策略, 包括混合精度训练、梯度累加和梯度截断等, 用户可以根据需求选择合适的训练策略.
优化器封装还定义了一套标准的参数更新流程, 用户可以基于这一套流程, 在同一套代码里, 实现不同训练策略的切换. 如果想了解更多, 可以参考 [MMEngine 优化器封装文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

MMSegmenetation 训练模型也是使用优化器封装来优化参数, 以下是 MMSegmentation 中常用的使用方法:

#### 配置 PyTorch 支持的优化器

OpenMMLab 2.0 支持 PyTorch 原生所有优化器, 参考[这里](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md#%E7%AE%80%E5%8D%95%E9%85%8D%E7%BD%AE).
在配置文件中设置训练时 `Runner` 所使用的优化器, 需要定义 `optim_wrapper`, 例如配置使用 SGD 优化器:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
```

#### `paramwise_cfg` 参数

在模型训练中, 如果想在优化器里为不同参数设置优化策略, 例如设置不同的学习率、权重衰减, 可以通过设置 `paramwise_cfg` 来实现.

例如, 在使用 ViT 作为模型骨干网络进行训练时, 优化器中设置了权重衰减 (weight decay), 但对 position embedding, layer normalization 和 class token 参数需要关掉 weight decay, `optim_wrapper` 的配置[如下](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py#L15-L27):

```python
optimizer = dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
```

其中 `decay_mult` 指的是对应参数的权重衰减的系数. 关于更多 `paramwise_cfg` 的使用可以参考 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

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
