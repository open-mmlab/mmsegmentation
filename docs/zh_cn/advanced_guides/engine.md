# 训练引擎

## 钩子 (Hook)

### 介绍

钩子机制被广泛地使用在如今的 OpenMMLab 开源算法库中，被 `Runner` 插入，整个训练过程的生命周期可以被容易地管理。您可以通过 [相关文章](https://www.calltutors.com/blog/what-is-hook/) 了解更多。

钩子仅会在被注册到 runner 后才会生效，现在，钩子主要被分成两类:

- 默认钩子 (default hooks)

被 runner 默认注册的钩子。通常来说，它们实现了一些基础的功能，而且有默认的优先级，用户不需要修改它们的优先级。

- 定制钩子 (custom hooks)

定制钩子通过 `custom_hooks` 被注册。通常来说，它们是实现增强功能的，优先级需要在配置文件里设置。如果用户没有设置，则会被默认设置为 `NORMAL`。

**优先级列表**:

|      级别       | 值  |
| :-------------: | :-: |
|     HIGHEST     |  0  |
|    VERY_HIGH    | 10  |
|      HIGH       | 30  |
|  ABOVE_NORMAL   | 40  |
| NORMAL(default) | 50  |
|  BELOW_NORMAL   | 60  |
|       LOW       | 70  |
|    VERY_LOW     | 90  |
|     LOWEST      | 100 |

级别决定了这些钩子的执行顺序。为了方便代码调试，在训练前，会在日志里打印出钩子在每个训练阶段的执行顺序。

### 默认钩子

下面的常用的钩子已经被 MMEngine 的 [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1759)　给 [默认](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/schedules/schedule_160k.py#L19-L25) 注册。

|                                                           钩子                                                            |                                             使用方法                                             |      优先级       |
| :-----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :---------------: |
|            [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)            |                                    记录 iteration 花费的时间.                                    |    NORMAL (50)    |
|               [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)                | 从 `Runner` 里不同的组件中收集日志记录，并将其输出到终端， JSON 文件，tensorboard，wandb 等下游. | BELOW_NORMAL (60) |
|       [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py)       |                          更新优化器里面的一些超参数，例如学习率的动量.                           |     LOW (70)      |
|           [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)            |                                   规律性地保存 checkpoint 文件                                   |   VERY_LOW (90)   |
|        [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)        |                                确保分布式采样器 shuffle 是打开的                                 |    NORMAL (50)    |
| [SegVisualizationHook](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/visualization/local_visualizer.py) |                                 可视化验证和测试过程里的预测结果                                 |    NORMAL (50)    |

注意: `SegVisualizationHook`　是在 MMSegmentation 里被实现的钩子，之后会专门介绍。

### MMEngine 里实现的定制钩子

一些钩子已经被实现在 MMEngine 里面了，它们是:

|                                                         钩子                                                          |                                         使用方法                                         |    优先级    |
| :-------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :----------: |
|                [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)                 |             在模型训练时使用指数滑动平均 (Exponential Moving Average, EMA).              | NORMAL (50)  |
|         [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)         |                        在训练时释放所有没有被缓存占用的 GPU 显存.                        | NORMAL (50)  |
|        [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py)         | 在每个 训练 Epoch 结束时同步模型 buffer 里的参数例如 BN 里的 running_mean 和 running_var | NORMAL (50)  |
| [NaiveVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py) |                               在测试时展示或写出预测结果.                                | LOWEST (100) |

### MMSegmentation 里实现的定制钩子

在 MMSegmentation 里有一个已经被实现的钩子，它就是 [SegVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py), 主要用来在验证和测试时可视化预测结果。

`SegVisualizationHook` 是这样实现的:

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

关于可视化更多的细节可以在查看[这里](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/visualization.md).

如果钩子已经被 MMEngine 或 MMSegmentation 实现，你可以直接修改配置文件来使用它们。例如在第 500 个 iteration 时开始使用 EMA:

```python
custom_hooks = [
    dict(type='EMAHook', start_iters=500, priority='NORMAL')
]
```

## 优化器

我们将通过三个部分介绍优化器的内容: 优化器(Optimizer)，优化器包装(Optimizer wrapper) 和构造器(Constructor)。

### 优化器

#### 定制 PyTorch 支持的优化器

我们已经支持了 PyTorch 所有的优化器，参考 `mmengine/optim/optimizer/builder.py`。请修改配置文件里的 `optimizer` 来使用和修改它们。

例如，如果你想使用 SGD，可以如下操作:

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

如果要修改模型的学习率，仅需要修改配置文件里 optimizer 的 `lr`。你还可以直接根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 设置其他参数。

例如，如果你想使用 `Adam`, 在 PyTorch 里它的设置是 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)`，可以这样修改配置文件:

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

#### 逐个参数化的配置

一些模型里不同的参数可能会有不同的优化策略，例如每个 layer 没有 bias 而且 BatchNorm 层没有 weight decay。为了实现这样的需求，我们可以在优化器里使用 `paramwise_cfg`。

例如，在 ViT 里面，我们不想在主干网络里的 position embedding & layer norm 实现 weight decay．我们可以采用如下的 [配置文件](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py#L15-L27)
来操作相关的参数  `pos_embed`, `cls_token` 和 `norm`:

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

### 优化器包装

除了实现 PyTorch 优化器的基本功能外，我们还提供了一些额外的功能，例如梯度裁剪(gradient clipping)，梯度累积(gradient accumulation)，自动混合精度训练 (automatic mixed precision training)。更多细节请参考 [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py)。

#### 梯度裁剪

目前我们在 `optim_wrapper` 里面支持 `clip_grad` 这个选项，更多细节你可以参考 [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) 和　[PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) 。例如：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
	optimizer=optimizer,
    clip_grad=dict(
        max_norm=0.2,
        norm_type=2))
# norm_type: 这里的 norm_type 是 L2 范数.
```

如果 `clip_grad` 不是 None, 它里面的字段就是 `torch.nn.utils.clip_grad.clip_grad_norm_()` 里面的参数。

#### 梯度累积

当没有足够的计算资源时， batch size 可以被设置地比较小，这将会降低模型的性能，梯度累积可以被用来解决这个问题。

有这样的例子:

```python
train_dataloader = dict(batch_size=4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=4)
```

它的意思是在训练时，每 4 个 iteration 才进行一次梯度反传,这等价于:

```python
train_dataloader = dict(batch_size=16)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=1)
```

#### 自动混合精度 (Automatic mixed precision, AMP) 训练

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

`AmpOptimWrapper` 里 `loss_scale` 默认设置为 `dynamic`。

### 构造器

构造器用来构建优化器，优化器包装和定制化模型不同层的超参数。配置文件中的 `optim_wrapper` 的 `paramwise_cfg` 用来控制这个定制化。

#### MMSegmentation 实现的构造器

- [LearningRateDecayOptimizerConstructor](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py#L104)

`LearningRateDecayOptimizerConstructor` 对主干网络的不同层设置不同的学习率。现在，这个优化构造器仅用于 ConvNeXt, BEiT 和 MAE。

示例:

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

注意: `paramwise_cfg` 将被忽略，它可以被写成 `paramwise_cfg=dict()`.
