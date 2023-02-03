# Customize Runtime Settings

## Customize hooks

### Step 1: Implement a new hook

MMEngine has implemented commonly used [hooks](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md) for training and test,
When users have requirements for customization, they can follow examples below.
For example, if some hyper-parameter of the model needs to be changed when model training, we can implement a new hook for it:

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
        # acquire this model when it is in a wrapper
        if is_model_wrapper(runner.model):
          model = runner.model.module
        model.hyper_parameter = self.a * cur_iter + self.b
```

### Step 2: Import a new hook

The module which is defined above needs to be imported into main namespace first to ensure being registered.
We assume `NewHook` is implemented in `mmseg/engine/hooks/new_hook.py`, there are two ways to import it:

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
The execution priority of hooks at the same place of `Runner` can be referred [here](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md#built-in-hooks),
Default priority of customized hook is `NORMAL`.

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

## Customize optimizer

### Step 1: Implement a new optimizer

We recommend the customized optimizer implemented in `mmseg/engine/optimizers/my_optimizer.py`. Here is an example of a new optimizer `MyOptimizer` which has parameters `a`, `b` and `c`:

```python
from mmseg.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)
```

### Step 2: Import a new optimizer

The module which is defined above needs to be imported into main namespace first to ensure being registered.
We assume `MyOptimizer` is implemented in `mmseg/engine/optimizers/my_optimizer.py`, there are two ways to import it:

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

### Step 1: Implement a new optimizer constructor

Optimizer constructor is used to create optimizer and optimizer wrapper for model training, which has powerful functions like specifying learning rate and weight decay for different model layers.
Here is an example for a customized optimizer constructor.

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
It can also be used as base class of new optimizer constructor.

### Step 2: Import a new optimizer constructor

The module which is defined above needs to be imported into main namespace first to ensure being registered.
We assume `MyOptimizerConstructor` is implemented in `mmseg/engine/optimizers/my_optimizer_constructor.py`, there are two ways to import it:

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
