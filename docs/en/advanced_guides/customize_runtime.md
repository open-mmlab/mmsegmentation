# Customize Runtime Settings

## Customize hooks

### Step 1: Create a new hook

MMEngine has implemented commonly used [hooks](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md) for training and test,
When users have requirements for customization, they can follow examples below.
For example, if they want to modify the value of hyperparameter `model.hyper_paramete`, aiming at making it change with training iteration number:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


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
        # 当模型被包在 wrapper 里时获取这个模型
        if is_model_wrapper(runner.model):
          model = runner.model.module
        model.hyper_parameter = self.a * cur_iter + self.b
```

### Step 2: Import a new hook

The module which is defined above needs to be imported into main namespace first to ensure being detected by program.
Assume NewHook is in `mmseg/engine/hooks/new_hook.py`, there are two ways to implement it:

- Import it by modifying `mmseg/engine/hooks/__init__.py`.
  Modules should be imported in `mmseg/engine/hooks/__init__.py` thus these new modules can be found and added by registry.

```python
from .new_hook import NewHook

__all__ = [..., NewHook]
```

- Import it manually by `custom_imports` in config file.

```python
custom_imports = dict(imports=['mmseg.engine.hooks.new_hook'], allow_failed_imports=False)
```

### Step 3: Modify config file

Users can set and use customized hooks in training and test followed methods below.
The priorities of different hooks which are located in the same place can be referred [here](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md#built-in-hooks),
Default priority of customized hook is `NORMAL`.

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

## Customize optimizer

### Step 1: Create a new optimizer

If users want to add a new optimizer `MyOptimizer` which has parameters `a`, `b` and `c`. We recommend implementing it in `mmseg/engine/optimizers/my_optimizer.py`:

```python
from mmseg.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)
```

### Step 2: Import a new optimizer

The module which is defined above needs to be imported into main namespace first to ensure being detected by program.
Assume `MyOptimizer` is in `mmseg/engine/optimizers/my_optimizer.py`, there are two ways to implement it:

- Import it by modifying `mmseg/engine/optimizers/__init__.py`.
  Modules should be imported in `mmseg/engine/optimizers/__init__.py` thus these new modules can be found and added by registry.

```python
from .my_optimizer import MyOptimizer
```

- Import it manually by `custom_imports` in config file.

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

### Step 3: Modify config file

Then it needs to modify `optimizer` in `optim_wrapper` of config file, if users want to use customized `MyOptimizer`, it can be modified as:

```python
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='MyOptimizer',
                                    a=a_value, b=b_value, c=c_value),
                     clip_grad=None)
```

## Customize optimizer constructor

### Step 1: Create a new optimizer constructor

Constructor can be used to create optimizer, optimizer wrapper and customized hyperparameters of different layers in model network. The optimizer of some models would adjust with specified parameters such as weight decay of BatchNorm layer.
Users can set different optimization policy on different parameters of model by customized optimizer constructor.

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer
```

Default optimizer constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L19).
It can also be used as template of new optimizer constructor.

### Step 2: Import a new optimizer constructor

The module which is defined above needs to be imported into main namespace first to ensure being detected by program.
Assume `MyOptimizerConstructor` is in `mmseg/engine/optimizers/my_optimizer_constructor.py`, there are two ways to implement it:

- Import it by modifying `mmseg/engine/optimizers/__init__.py`.
  Modules should be imported in `mmseg/engine/optimizers/__init__.py` thus these new modules can be found and added by registry.

```python
from .my_optimizer_constructor import MyOptimizerConstructor
```

- Import it manually by `custom_imports` in config file.

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer_constructor'], allow_failed_imports=False)
```

### Step 3: Modify config file

Then it needs to modify `constructor` in `optim_wrapper` of config file, if users want to use customized `MyOptimizerConstructor`, it can be modified as:

```python
optim_wrapper = dict(type='OptimWrapper',
                     constructor='MyOptimizerConstructor',
                     clip_grad=None)
```
