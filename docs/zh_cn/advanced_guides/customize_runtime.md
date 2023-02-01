# 自定义运行设定

## 实现自定义钩子

### Step 1: 创建一个新的钩子

MMEngine 已实现了训练和测试常用的[钩子](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md),
当有定制化需求时, 可以按照如下示例实现适用于自身训练需求的钩子, 例如想修改一个超参数 `model.hyper_paramete` 的值, 让它随着训练迭代次数而变化:

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

### Step 2: 导入一个新的钩子

为了让上面定义的模块可以被执行的程序发现, 这个模块需要先被导入主命名空间 (main namespace) 里面,
假设 NewHook 在 `mmseg/engine/hooks/new_hook.py` 里面, 有两种方式去实现它:

- 修改 `mmseg/engine/hooks/__init__.py` 来导入它.
  新定义的模块应该在 `mmseg/engine/hooks/__init__.py` 里面导入, 这样注册器可以发现并添加这个新的模块:

```python
from .new_hook import NewHook

__all__ = [..., NewHook]
```

- 在配置文件里使用 custom_imports 来手动导入它.

```python
custom_imports = dict(imports=['mmseg.engine.hooks.new_hook'], allow_failed_imports=False)
```

### Step 3: 修改配置文件

可以按照如下方式, 在训练或测试中配置并使用自定义的钩子. 不同钩子在同一位点的优先级可以参考[这里](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md#%E5%86%85%E7%BD%AE%E9%92%A9%E5%AD%90), 自定义钩子如果没有指定优先, 默认是 `NORMAL`.

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

## 实现自定义优化器

### Step 1: 创建一个新的优化器

如果增加一个叫作 `MyOptimizer` 的优化器, 它有参数 `a`, `b` 和 `c`. 推荐在 `mmseg/engine/optimizers/my_optimizer.py` 文件中实现

```python
from mmseg.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)
```

### Step 2: 导入一个新的优化器

为了让上面定义的模块可以被执行的程序发现, 这个模块需要先被导入主命名空间 (main namespace) 里面,
假设 `MyOptimizer` 在 `mmseg/engine/optimizers/my_optimizer.py` 里面, 有两种方式去实现它:

- 修改 `mmseg/engine/optimizers/__init__.py` 来导入它.
  新定义的模块应该在 `mmseg/engine/optimizers/__init__.py` 里面导入, 这样注册器可以发现并添加这个新的模块:

```python
from .my_optimizer import MyOptimizer
```

- 在配置文件里使用 `custom_imports` 来手动导入它.

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

### Step 3: 修改配置文件

随后需要修改配置文件 `optim_wrapper` 里的 `optimizer` 参数, 如果要使用你自己的优化器 `MyOptimizer`, 字段可以被修改成:

```python
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='MyOptimizer',
                                    a=a_value, b=b_value, c=c_value),
                     clip_grad=None)
```

## 实现自定义优化器封装构造器

### Step 1: 创建一个新的优化器封装构造器

构造器可以用来创建优化器, 优化器包, 以及自定义模型网络不同层的超参数. 一些模型的优化器可能会根据特定的参数而调整, 例如 BatchNorm 层的 weight decay. 使用者可以通过自定义优化器构造器来精细化设定不同参数的优化策略.

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer
```

默认的优化器构造器在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L19) 被实现, 它也可以用来作为新的优化器构造器的模板.

### Step 2: 导入一个新的优化器封装构造器

为了让上面定义的模块可以被执行的程序发现, 这个模块需要先被导入主命名空间 (main namespace) 里面, 假设 `MyOptimizerConstructor` 在 `mmseg/engine/optimizers/my_optimizer_constructor.py` 里面, 有两种方式去实现它:

- 修改 `mmseg/engine/optimizers/__init__.py` 来导入它.
  新定义的模块应该在 `mmseg/engine/optimizers/__init__.py` 里面导入, 这样注册器可以发现并添加这个新的模块:

```python
from .my_optimizer_constructor import MyOptimizerConstructor
```

- 在配置文件里使用 `custom_imports` 来手动导入它.

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer_constructor'], allow_failed_imports=False)
```

### Step 3: 修改配置文件

随后需要修改配置文件 `optim_wrapper`  里的 `constructor` 参数, 如果要使用你自己的优化器封装构造器 `MyOptimizerConstructor`, 字段可以被修改成:

```python
optim_wrapper = dict(type='OptimWrapper',
                     constructor='MyOptimizerConstructor',
                     clip_grad=None)
```
