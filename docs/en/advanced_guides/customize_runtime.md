# Customize Runtime Settings

<!-- TOC -->

- [Customize Runtime](#customize-runtime)
  - [Loop](#loop)
  - [Hook](#hook)
    - [Step 1: Create a new hook](#step-1-create-a-new-hook)
    - [Step 2: Import the new hook](#step-2-import-the-new-hook)
    - [Step 3: Modify the config](#step-3-modify-the-config)
    - [Modify default runtime hooks](#modify-default-runtime-hooks)
  - [Optimizer](#optimizer)
    - [Optimizer Wrapper](#optimizer-wrapper)
    - [Constructor](#constructor)
    - [Customize self-implemented optimizer](#customize-self-implemented-optimizer)
    - [Customize optimizer constructor](#customize-optimizer-constructor)
    - [Additional settings](#additional-settings)
  - [Scheduler](#scheduler)

<!-- /TOC -->

It is necessary to design an engine to dispatch various modules related with training and inference processes of OpenMMLab codebases. In MMEngine, `Runner` is basic class utilized for dispatch of OpenMMLab codebases in their training process.

In this tutorial, we will introduce some methods about how to customize runtime settings for the project.

## Loop

MMEngine defines several [basic loops](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) such as `EpochBasedTrainLoop`, `IterBasedTrainLoop`, `ValLoop` and `TestLoop`.

`Loop` means the workflow of training, validation or testing. We use `train_cfg`, `val_cfg` and `test_cfg` in config file to build `Loop`.

E.g.:

```python
# Use IterBasedTrainLoop to train 200 epochs.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
```

MMEngine defines several [basic loops](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py). Users could implement customized loops if the defined loops are not satisfied.

## Hook

MMSegmentation would register some hooks which are commonly used by [defualt_hooks](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/schedules/schedule_160k.py#L19-L25) in config files.

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

Before learning to create your customized hooks, it is recommended to learn the basic concept of hooks in file [engine.md](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/advanced_guides/engine.md).

### Step 1: Create a new hook

Depending on your intention of this hook, you need to implement corresponding functions according to the hook point of your expectation.

For example, if you want to modify the value of a hyper-parameter according to the training iter and two other hyper-parameters after every train iter, you could implement a hook like:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS
from mmseg.utils import get_model


@HOOKS.register_module()
class NewHook(Hook):
    """Docstring for NewHook.
    """

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        cur_iter = runner.iter
        # Get model if the input model is a model wrapper
        if is_model_wrapper(runner.model):
          model = runner.model.module
        model.hyper_parameter = self.a * cur_iter + self.b
```

### Step 2: Import the new hook

Then we need to ensure `NewHook` imported. Assuming `NewHook` is in `mmseg/engine/hooks/new_hook.py`, modify `mmseg/engine/hooks/__init__.py` as below

```python
...
from .new_hook import NewHook

__all__ = [..., NewHook]
```

### Step 3: Modify the config

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook as below:

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

By default, the hook's priority is set as `NORMAL` during registration.

### Modify default runtime hooks

There are some common hooks that are not registered through `custom_hooks`, they are:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

Here we reveals how what we can do with `logger` and `checkpoint`.

#### Checkpoint config

The MMEngine runner will use `checkpoint_config` to initialize [`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19).

```python
checkpoint = dict(interval=1)
```

The users could set `max_keep_ckpts` to only save only small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`. More details of the arguments are [here](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html?highlight=CheckpointHook).

#### Log config

The `LoggerHook` is designed to collect logs from different components of `Runner` and write them to terminal, JSON file, tensorboard and wandb .etc.

```python
logger_hook_cfg = dict(interval=20)
```

In latest 1.x MMSegmentation, some `LoggerHook` such as `TextLoggerHook`, `WandbLoggerHook` and `TensorboardLoggerHook` would no longer be kept.
Instead, MMEngine utilizes `LogProcessor` to format log information collected by `MessageHub` in `runner.message_hub`, and `WandbVisBackend` or `TensorboardVisBackend` in `runner.visualizer`.

## Optimizer

Before customizing the optimizer config, it is recommended to learn the basic concept of optimizer in [engine.md](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/advanced_guides/engine.md).

Here is an example of SGD optimizer:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

We support all optimizers of PyTorch. For more details, please refer to [MMEngine optimizer document](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

### Optimizer Wrapper

Optimizer wrapper provides a unified interface for single precision training and automatic mixed precision training with different hardware. Here is an example of `optim_wrapper` setting:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

Besides, if you want to apply automatic mixed precision training, you could modify the config above like:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

The default setting of `loss_scale` of `AmpOptimWrapper` is `dynamic`.

### Constructor

The constructor aims to build optimizer, optimizer wrapper and customize hyper-parameters of different layers. The key `paramwise_cfg` of `optim_wrapper` in configs controls this customization.

The example and detailed information can be found in [MMEngine optimizer document](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

Besides, We could use `custom_keys` to set different hyper-parameters of different modules.

Here is the `optim_wrapper` example of MAE. The config below sets weight decay multiplication to be 0 of `pos_embed`, `mask_token`, `norm` modules. During training, the weight decay of these modules will be `weight_decay * decay_mult`.

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

Furthermore, for some specific settings, we could use int type arguments to control the `weight_decay` and `lr_scale` of parameters. For example, here is an example config of SimCLR:

```python
optimizer = dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')
```

In `AdamW` optimizer example above, we have `num_layers` and `decay_type` to decide specific `weight_decay` and `lr_scale` value of certain model parameters.

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as following.

Assume you want to add a optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmseg/engine/optimizers`.
And then implement the new optimizer in a file, e.g., in `mmseg/engine/optimizers/my_optimizer.py`:

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two options to achieve it.

- Modify `mmseg/engine/optimizers/__init__.py` to import it.

  The newly defined module should be imported in `mmseg/engine/optimizers/__init__.py` so that the registry will
  find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

The module `mmseg.engine.optimizers.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported.
`mmseg.engine.optimizers.my_optimizer.MyOptimizer` **cannot** be imported directly.

Actually users can use a totally different file directory structure using this importing method, as long as the module root can be located in `PYTHONPATH`.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed to

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer constructor.

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

The default optimizer constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L19), which could also serve as a template for new optimizer constructor.

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer constructor (e.g., set parameter-wise learning rates) or hooks. We list some common settings that could stabilize the training or accelerate the training. Feel free to create PR, issue for more settings.

- __Use gradient clip to stabilize training__:
  Some models need gradient clip to clip the gradients to stabilize the training process. An example is as below:

  ```python
  optim_wrapper = dict(
      _delete_=True,
      type='OptimWrapper',
      grad_clip=dict(max_norm=1, norm_type=2))
  ```

  If your config inherits the base config which already sets the `optim_wrapper`, you might need `_delete_=True` to override the unnecessary settings. See the [config documentation](https://mmsegmentation.readthedocs.io/en/latest/config.html) for more details.

## Customize training schedules

Before customizing the scheduler config, it is recommended to learn the basic concept of scheduler in [MMEngine document](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md).

Here is an example of scheduler:

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

**Note:** When you change the `max_iters` in `train_cfg`, make sure that the args in `param_scheduler` are modified simultanuously.
