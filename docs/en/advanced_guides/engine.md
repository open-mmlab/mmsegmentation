# Engine

<!-- TOC -->

- [Engine](#engine)
  - [Hook](#hook)
    - [Introduction](#introduction)
    - [Default hooks](#default-hooks)
    - [Common Hooks implemented in MMEngine](#common-hooks-implemented-in-mmengine)
    - [Hooks implemented in MMSegmentation](#hooks-implemented-in-mmsegmentation)
  - [Optimizer](#optimizer)
    - [Optimizer](#optimizer-1)
      - [Customize optimizer supported by PyTorch](#customize-optimizer-supported-by-pytorch)
      - [Parameter-wise configuration](#parameter-wise-configuration)
    - [Optimizer wrapper](#optimizer-wrapper)
      - [Gradient clipping](#gradient-clipping)
      - [Gradient accumulation](#gradient-accumulation)
      - [Automatic mixed precision(AMP) training](#automatic-mixed-precisionamp-training)
    - [Constructor](#constructor)
      - [Constructors implemented in MMSegmentation](#constructors-implemented-in-mmsegmentation)

<!-- /TOC -->

## Hook

### Introduction

The hook mechanism is widely used in the OpenMMLab open-source algorithm library. Inserted in the `Runner`, the entire life cycle of the training process can be managed easily. You can learn more about the hook through [related article](https://www.calltutors.com/blog/what-is-hook/).

Hooks only work after being registered into the runner. At present, hooks are mainly divided into two categories:

- default hooks

Those hooks are registered by the runner by default. Generally, they fulfill some basic functions, and have default priority, you don't need to modify the priority.

- custom hooks

The custom hooks are registered through custom_hooks. Generally, they are hooks with enhanced functions. The priority needs to be specified in the configuration file. If you do not specify the priority of the hook, it will be set to `NORMAL` by default.

**Priority list**:

|      Level      | Value |
| :-------------: | :---: |
|     HIGHEST     |   0   |
|    VERY_HIGH    |  10   |
|      HIGH       |  30   |
|  ABOVE_NORMAL   |  40   |
| NORMAL(default) |  50   |
|  BELOW_NORMAL   |  60   |
|       LOW       |  70   |
|    VERY_LOW     |  90   |
|     LOWEST      |  100  |

The priority determines the execution order of the hooks. Before training, the log will print out the execution order of the hooks at each stage to facilitate debugging.

### Default hooks

The following common hooks are already reigistered by [default](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/schedules/schedule_160k.py#L19-L25), which are implemented through [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1759) in MMEngine:

|                                                           Hooks                                                           |                                                         Usage                                                         |     Priority      |
| :-----------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :---------------: |
|            [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)            |                                         log the time spent during iteration.                                          |    NORMAL (50)    |
|               [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)                | collect logs from different components of `Runner` and write them to terminal, JSON file, tensorboard and wandb .etc. | BELOW_NORMAL (60) |
|       [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py)       |                     update some hyper-parameters in optimizer, e.g., learning rate and momentum.                      |     LOW (70)      |
|           [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)            |                                            save checkpoints periodically.                                             |   VERY_LOW (90)   |
|        [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)        |                                     ensure distributed Sampler shuffle is active.                                     |    NORMAL (50)    |
| [SegVisualizationHook](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/visualization/local_visualizer.py) |                             visualize validation and testing process prediction results.                              |    NORMAL (50)    |

Noted that `SegVisualizationHook` is hooks implemented in MMSegmentation, which would be introduced later.

### Common Hooks implemented in MMEngine

Some hooks have been already implemented in MMEngine, they are:

|                                                         Hooks                                                         |                                             Usage                                              |   Priority   |
| :-------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :----------: |
|                [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)                 |              apply Exponential Moving Average (EMA) on the model during training.              | NORMAL (50)  |
|         [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)         |            release all unoccupied cached GPU memory during the process of training.            | NORMAL (50)  |
|        [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py)         | synchronize model buffers such as running_mean and running_var in BN at the end of each epoch. | NORMAL (50)  |
| [NaiveVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py) |               Show or Write the predicted results during the process of testing.               | LOWEST (100) |

### Hooks implemented in MMSegmentation

One hook has been already implemented in MMSegmentation, it is [SegVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py),
used to visualize validation and testing process prediction results.

`SegVisualizationHook` is implemented as follows:

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
        # skip if it is in training process or self.draw is False
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

More details of visualization could be found [here](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/visualization.md).

If the hook is already implemented in MMEngine or MMSegmentation, you can directly modify the config to use the hook as below

```python
custom_hooks = [
    dict(type='MMEngineHook', a=a_value, b=b_value, priority='NORMAL')
]
```

such as using `EMAHook`, start_iters is 500:

```python
custom_hooks = [
    dict(type='EMAHook', start_iters=500)
]
```

## Optimizer

We will introduce Optimizer section through 3 different parts: Optimizer, Optimizer wrapper, and Constructor.

### Optimizer

#### Customize optimizer supported by PyTorch

We have already supported all the optimizers implemented by PyTorch, see `mmengine/optim/optimizer/builder.py`. To use and modify them, please change the `optimizer` field of config files.

For example, if you want to use SGD, the modification could be as the following.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer. You can also directly set other arguments according to the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with the setting like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch, the config should looks like:

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

#### Parameter-wise configuration

Some models may have some parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layer and the bias in each layer. To finely configure them, we can use the `paramwise_cfg` in optimizer.

For example, in ViT, we do not want to apply weight decay for position embedding & layer norm in backbone,
so we can use following [config file](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py#L15-L27)
to operate on related parameters `pos_embed`, `cls_token` and `norm`:

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

### Optimizer wrapper

Besides the basic function of PyTorch optimizers, we also provide some enhancement functions, such as gradient clipping, gradient accumulation, automatic mixed precision training, etc. Please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py) for more details.

#### Gradient clipping

Currently we support `clip_grad` option in `optim_wrapper`, and you can refer to [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) and [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)for more arguments . Here is an example:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
	optimizer=optimizer,
    clip_grad=dict(
        max_norm=0.2,
        norm_type=2))
# norm_type: type of the used p-norm, here norm_type is 2.
```

If `clip_grad` is not None, it will be the arguments of `torch.nn.utils.clip_grad.clip_grad_norm_()`.

#### Gradient accumulation

When there is not enough computation resource, the batch size can only be set to a small value, which may degrade the performance of model. Gradient accumulation can be used to solve this problem.

Here is an example:

```python
train_dataloader = dict(batch_size=4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
train_dataloader = dict(batch_size=16)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=1)
```

#### Automatic mixed precision(AMP) training

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

The default setting of `loss_scale` of `AmpOptimWrapper` is `dynamic`.

### Constructor

The constructor aims to build optimizer, optimizer wrapper and customize hyper-parameters of different layers. The key `paramwise_cfg` of `optim_wrapper` in configs controls this customization.

#### Constructors implemented in MMSegmentation

- [LearningRateDecayOptimizerConstructor](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py#L104)

`LearningRateDecayOptimizerConstructor` sets different learning rates for different layers of backbone. Note: Currently, this optimizer constructor is built for ConvNeXt, BEiT and MAE.

An example:

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

Note: `paramwise_cfg` will be ignored, and it can be written as  `paramwise_cfg=dict()`.
