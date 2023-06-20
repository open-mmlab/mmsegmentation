# Training Engine

MMEngine defined some [basic loop controllers](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) such as epoch-based training loop (`EpochBasedTrainLoop`), iteration-based training loop (`IterBasedTrainLoop`), standard validation loop (`ValLoop`), and standard testing loop (`TestLoop`).

OpenMMLab's algorithm libraries like MMSegmentation abstract model training, testing, and inference as `Runner` to handle. Users can use the default `Runner` in MMEngine directly or modify the `Runner` to meet customized needs. This document mainly introduces how users can configure existing running settings, hooks, and optimizers' basic concepts and usage methods.

## Configuring Runtime Settings

### Configuring Training Iterations

Loop controllers refer to the execution process during training, validation, and testing. `train_cfg`, `val_cfg`, and `test_cfg` are used to build these processes in the configuration file. MMSegmentation sets commonly used training iterations in `train_cfg` under the `configs/_base_/schedules` folder.
For example, to train for 80,000 iterations using the iteration-based training loop (`IterBasedTrainLoop`) and perform validation every 8,000 iterations, you can set it as follows:

```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
```

### Configuring Training Optimizers

Here's an example of a SGD optimizer:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
```

OpenMMLab supports all optimizers in PyTorch. For more details, please refer to the [MMEngine optimizer documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md).

It is worth emphasizing that `optim_wrapper` is a variable of `runner`, so when configuring the optimizer, the field to configure is the `optim_wrapper` field. For more information on using optimizers, see the [Optimizer](#Optimizer) section below.

### Configuring Training Parameter Schedulers

Before configuring the training parameter scheduler, it is recommended to first understand the basic concepts of parameter schedulers in the [MMEngine documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md).

Here's an example of a parameter scheduler. During training, a linearly changing learning rate strategy is used for warm-up in the first 1,000 iterations. After the first 1,000 iterations until the 16,000 iterations in the end, the default polynomial learning rate decay is used:

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

Note: When modifying the `max_iters` in `train_cfg`, make sure the parameters in the parameter scheduler `param_scheduler` are also modified accordingly.

## Hook

### Introduction

OpenMMLab abstracts the model training and testing process as `Runner`. Inserting hooks can implement the corresponding functionality needed at different training and testing stages (such as "before and after each training iter", "before and after each validation iter", etc.) in `Runner`. For more introduction on hook mechanisms, please refer to [here](https://www.calltutors.com/blog/what-is-hook).

Hooks used in `Runner` are divided into two categories:

- Default hooks:

They implement essential functions during training and are defined in the configuration file by `default_hooks` and passed to `Runner`. `Runner` registers them through the [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1780) method.

Hooks have corresponding priorities; the higher the priority, the earlier the runner calls them. If the priorities are the same, the calling order is consistent with the hook registration order.

It is not recommended for users to modify the default hook priorities. Please refer to the [MMEngine hooks documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md) to understand the hook priority definitions.

The following are the default hooks used in MMSegmentation:

|                                                          Hook                                                          |                                                          Function                                                          |     Priority      |
| :--------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: | :---------------: |
|          [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)           |                                          Record the time spent on each iteration.                                          |    NORMAL (50)    |
|              [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)              | Collect log records from different components in `Runner` and output them to terminal, JSON file, tensorboard, wandb, etc. | BELOW_NORMAL (60) |
|     [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py)      |                       Update some hyperparameters in the optimizer, such as learning rate momentum.                        |     LOW (70)      |
|          [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)          |                                              Regularly save checkpoint files.                                              |   VERY_LOW (90)   |
|      [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)       |                                     Ensure the distributed sampler shuffle is enabled.                                     |    NORMAL (50)    |
| [SegVisualizationHook](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/visualization/local_visualizer.py) |                                Visualize prediction results during validation and testing.                                 |    NORMAL (50)    |

MMSegmentation registers some hooks with essential training functions in `default_hooks`:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=32000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

All the default hooks mentioned above, except for `SegVisualizationHook`, are implemented in MMEngine. The `SegVisualizationHook` is a hook implemented in MMSegmentation, which will be introduced later.

- Modifying default hooks

We will use the `logger` and `checkpoint` in `default_hooks` as examples to demonstrate how to modify the default hooks in `default_hooks`.

(1) Model saving configuration

`default_hooks` uses the `checkpoint` field to initialize the [model saving hook (CheckpointHook)](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19).

```python
checkpoint = dict(type='CheckpointHook', interval=1)
```

Users can set `max_keep_ckpts` to save only a small number of checkpoints or use `save_optimizer` to determine whether to save optimizer information. More details on related parameters can be found [here](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html#checkpointhook).

(2) Logging configuration

The `LoggerHook` is used to collect log information from different components in `Runner` and write it to terminal, JSON files, tensorboard, wandb, etc.

```python
logger=dict(type='LoggerHook', interval=10)
```

In the latest 1.x version of MMSegmentation, some logger hooks (LoggerHook) such as `TextLoggerHook`, `WandbLoggerHook`, and `TensorboardLoggerHook` will no longer be used. Instead, MMEngine uses `LogProcessor` to handle the information processed by the aforementioned hooks, which are now in [`MessageHub`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/logging/message_hub.py#L17), [`WandbVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L324), and [`TensorboardVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L472).

Detailed usage is as follows, configuring the visualizer and specifying the visualization backend at the same time, here using Tensorboard as the visualizer's backend:

```python
# TensorboardVisBackend
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=[dict(type='TensorboardVisBackend')], name='visualizer')
```

For more related usage, please refer to [MMEngine Visualization Backend User Tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md).

- Custom hooks

Custom hooks are defined in the configuration through `custom_hooks`, and `Runner` registers them using the [`register_custom_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1820) method.

The priority of custom hooks needs to be set in the configuration file; if not, it will be set to `NORMAL` by default. The following are some custom hooks implemented in MMEngine:

|                                                  Hook                                                  |                                                               Usage                                                                |
| :----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
|         [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)         |                                    Use Exponential Moving Average (EMA) during model training.                                     |
| [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)  |                                  Release all GPU memory not occupied by the cache during training                                  |
| [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py) | Synchronize the parameters in the model buffer, such as `running_mean` and `running_var` in BN, at the end of each training epoch. |

The following is a use case for `EMAHook`, where the config file includes the configuration of the implemented custom hooks as members of the `custom_hooks` list.

```python
custom_hooks = [
    dict(type='EMAHook', start_iters=500, priority='NORMAL')
]
```

### SegVisualizationHook

MMSegmentation implemented [`SegVisualizationHook`](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/engine/hooks/visualization_hook.py#L17), which is used to visualize prediction results during validation and testing.
`SegVisualizationHook` overrides the `_after_iter` method in the base class `Hook`. During validation or testing, it calls the `add_datasample` method of `visualizer` to draw semantic segmentation results according to the specified iteration interval. The specific implementation is as follows:

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
        # If it's a training phase or self.draw is False, then skip it
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

For more details about visualization, you can check [here](../user_guides/visualization.md).

## Optimizer

In the previous configuration and runtime settings, we provided a simple example of configuring the training optimizer. This section will further detailly introduce  how to configure optimizers in MMSegmentation.

## Optimizer Wrapper

OpenMMLab 2.0 introduces an optimizer wrapper that supports different training strategies, including mixed-precision training, gradient accumulation, and gradient clipping. Users can choose the appropriate training strategy according to their needs. The optimizer wrapper also defines a standard parameter update process, allowing users to switch between different training strategies within the same code. For more information, please refer to the [MMEngine optimizer wrapper documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md).

Here are some common usage methods in MMSegmentation:

#### Configuring PyTorch Supported Optimizers

OpenMMLab 2.0 supports all native PyTorch optimizers, as referenced [here](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md).

To set the optimizer used by the `Runner` during training in the configuration file, you need to define `optim_wrapper` instead of `optimizer`. Below is an example of configuring an optimizer during training:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
```

#### Configuring Gradient Clipping

When the model training requires gradient clipping, you can configure it as shown in the following example:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer,
                        clip_grad=dict(max_norm=0.01, norm_type=2))
```

Here, `max_norm` refers to the maximum value of the gradient after clipping, and `norm_type` refers to the norm used when clipping the gradient. Related methods can be found in [torch.nn.utils.clip_grad_norm\_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

#### Configuring Mixed Precision Training

When mixed precision training is needed to reduce memory usage, you can use `AmpOptimWrapper`. The specific configuration is as follows:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

The default setting for `loss_scale` in [`AmpOptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/amp_optimizer_wrapper.py#L20) is `dynamic`.

#### Configuring Hyperparameters for Different Layers of the Model Network

In model training, if you want to set different optimization strategies for different parameters in the optimizer, such as setting different learning rates, weight decay, and other hyperparameters, you can achieve this by setting `paramwise_cfg` in the `optim_wrapper` of the configuration file.

The following config file uses the [ViT `optim_wrapper`](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py#L15-L27) as an example to introduce the use of `paramwise_cfg` parameters. During training, the weight decay parameter coefficients for the `pos_embed`, `mask_token`, and `norm` modules are set to 0. That is, during training, the weight decay for these modules will be changed to `weight_decay * decay_mult`=0.

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

Here, `decay_mult` refers to the weight decay coefficient for the corresponding parameters. For more information on the usage of `paramwise_cfg`, please refer to the [MMEngine optimizer wrapper documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md).

### Optimizer Wrapper Constructor

The default optimizer wrapper constructor [`DefaultOptimWrapperConstructor`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L19) builds the optimizer used in training based on the input `optim_wrapper` and `paramwise_cfg` defined in the `optim_wrapper`. When the functionality of [`DefaultOptimWrapperConstructor`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L19) does not meet the requirements, you can customize the optimizer wrapper constructor to implement the configuration of hyperparameters.

MMSegmentation has implemented the [`LearningRateDecayOptimizerConstructor`](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py#L104), which can decay the learning rate of model parameters in the backbone networks of ConvNeXt, BEiT, and MAE models during training according to the defined decay ratio (`decay_rate`). The configuration in the configuration file is as follows:

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

The purpose of `_delete_=True` is to ignore the inherited configuration in the OpenMMLab Config. In this code snippet, the inherited `optim_wrapper` configuration is ignored. For more information on `_delete_` fields, please refer to the [MMEngine documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md#delete-key-in-dict).
