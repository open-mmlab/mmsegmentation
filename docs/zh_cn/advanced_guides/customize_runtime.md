# 自定义运行设定

<!-- TOC -->

- [自定义优化设定](#自定义优化设定)
  - [循环控制器](#循环控制器)
  - [钩子](#钩子)
    - [Step 1: 创建一个新的钩子](#step-1-创建一个新的钩子)
    - [Step 2: 导入一个新的钩子](#step-2-导入一个新的钩子)
    - [Step 3: 修改配置文件](#step-3-修改配置文件)
    - [修改默认的钩子](#修改默认的钩子)
  - [优化器](#优化器)
    - [优化器包 (Optimizer wrapper)](<#优化器包-(Optimizer-wrapper)>)
    - [构造器](#构造器)
    - [自定义新的优化器](#自定义新的优化器)
    - [自定义优化器构造器](#自定义优化器构造器)
    - [额外的设置](#额外的设置)
  - [自定义参数调度器](#自定义参数调度器)

<!-- /TOC -->

OpenMMLab 代码库需要设计一个引擎去调度训练和推理过程的各个模块, 在 MMEngine 里面抽象出了 `Runner` (执行器) 来负责通用的算法模型的训练、测试、推理任务。用户一般可以直接使用 MMEngine 中的默认执行器，也可以对执行器进行修改以满足定制化需求。

## 循环控制器

MMEngine 定义了一些 [基础循环控制器](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) 例如 `基于轮次的训练循环 (EpochBasedTrainLoop)`, `基于迭代次数的训练循环 (IterBasedTrainLoop)`, `标准的验证循环 (ValLoop)` and `标准的测试循环 (TestLoop)`.

`循环控制器`  指的是训练, 验证和测试时的执行流程. 我们在配置文件里面使用 `train_cfg`, `val_cfg` 和 `test_cfg` 来构建 `Loop`.

例如:

```python
# 使用基于迭代次数的训练循环 (IterBasedTrainLoop)去训练 80000个迭代次数.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
```

## 钩子

MMSegmentation 会在 [`defualt_hooks`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/schedules/schedule_160k.py#L19-L25) 里面注册一些常用的钩子:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

在学习如何自定义钩子之前, 推荐先学习 [engine.md](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/advanced_guides/engine.md) 里面关于钩子的基本概念.

### Step 1: 创建一个新的钩子

你需要根据自己的需求去实现具有对应功能的钩子, 例如, 如果你想修改一个超参数 `model.hyper_paramete` 的值, 让它随着训练迭代次数而变化, 你可以实现如下的钩子:

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
        # 当模型被包在 wrapper 里时获取这个模型
        if is_model_wrapper(runner.model):
          model = runner.model.module
        model.hyper_parameter = self.a * cur_iter + self.b
```

### Step 2: 导入一个新的钩子

随后我们需要确保 `NewHook` 被导入. 假设 `NewHook` 在 `mmseg/engine/hooks/new_hook.py` 里面, 我们需要在 `mmseg/engine/hooks/__init__.py` 修改如下:

```python
...
from .new_hook import NewHook

__all__ = [..., NewHook]
```

### Step 3: 修改配置文件

你可以这样设定配置文件里的钩子， 在注册时．默认的优先级是 `NORMAL`.

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

### 修改默认的钩子

以 `default_hooks` 里面的 `logger` 和 `checkpoint` 为例, 我们来介绍如何修改 `default_hooks`中默认的钩子.

#### 模型保存配置

MMEngine 执行器将使用 `checkpoint` 来初始化 [`模型保存钩子 (CheckpointHook)`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19):

```python
checkpoint = dict(interval=1)
```

用户可以设置 `max_keep_ckpts` 来只保存少量的检查点或者用 `save_optimizer` 来决定是否保存 optimizer 的信息. 更多相关参数的细节可以参考[这里](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html?highlight=CheckpointHook).

#### 日志配置

`日志钩子 (LoggerHook)` 被用来收集`执行器 (Runner)`里面不同组件的日志信息然后写入终端, JSON 文件, tensorboard 和 wandb 等地方.

```python
logger_hook_cfg = dict(interval=20)
```

在最新的 1.x 版本的 MMSegmentation 里面, 一些日志钩子 (LoggerHook) 例如 `TextLoggerHook`, `WandbLoggerHook` and `TensorboardLoggerHook` 将不再被使用.
作为替代, MMEngine 使用 `LogProcessor` 来处理上述钩子处理的信息，它们现在在 [`MessageHub`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/logging/message_hub.py#L17), [`WandbVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L324) 和 [`TensorboardVisBackend`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py#L472) 里面.

## 优化器

在自定义优化器配置文件之前, 推荐先学习 [engine.md](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/advanced_guides/engine.md) 里面关于优化器的基本概念.
这里是一个 SGD 优化器的例子:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

我们支持 PyTorch 里面所有的优化器, 更多细节可以参考 [MMEngine 优化器文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

### 优化器包 (Optimizer wrapper)

优化器包 (Optimizer wrapper) 提供一个统一的在不同硬件上的接口. 下面是一个 `optim_wrapper` 的例子:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

除此之外, 如果你想应用混合精度训练, 你可以将上述配置文件修改如下:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

`AmpOptimWrapper` 中 `loss_scale` 的默认设置是 `dynamic`.

### 构造器

构造器可以用来创建优化器, 优化器包, 以及自定义模型网络不同层的超参数. 后者可以由配置文件里 `optim_wrapper` 中的 `paramwise_cfg` 来控制.

相关示例和详细信息可以在 [MMEngine 优化器文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md) 里面查到.

除此以外, 我们可以使用 `custom_keys` 来设置不同模块的超参数.

这里是 MAE `optim_wrapper`　的例子, 下面的配置文件是在将 `pos_embed`, `mask_token`, `norm` 模块的 weight decay multiplication 设置成 0. 在训练时, 这些模块的 weight decay 将被变为 `weight_decay * decay_mult`.

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

### 自定义新的优化器

#### 1. 定义一个新的优化器

一个定制化的优化器可以被如下定义:

假设你想增加一个叫作 `MyOptimizer` 的优化器, 它有参数 `a`, `b` 和  `c`.
你需要创建一个叫做　`mmseg/engine/optimizers` 的新的文件夹路径, 然后在 `mmseg/engine/optimizers/my_optimizer.py` 里面创建一个新的优化器.
然后在文件里面实现这个优化器, 例如:

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. 将优化器增加到注册器里

To find the above module defined above, this module should be imported into the main namespace at first. There are two options to achieve it.
为了让上面定义的模块可以被执行的程序发现, 这个模块需要先被导入主命名空间 (main namespace) 里面, 有两种方式去实现它:

- 修改 `mmseg/engine/optimizers/__init__.py` 来导入它.

  新定义的模块应该在 `mmseg/engine/optimizers/__init__.py` 里面导入, 这样注册器可以发现并添加这个新的模块:

```python
from .my_optimizer import MyOptimizer
```

- 在配置文件里使用 `custom_imports` 来手动导入它.

```python
custom_imports = dict(imports=['mmseg.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

`mmseg.engine.optimizers.my_optimizer` 模块将在程序开始前被导入然后 `MyOptimizer` 类将会被自动注册.
需要注意只有包括 `MyOptimizer` 类的包才应该被导入.
`mmseg.engine.optimizers.my_optimizer.MyOptimizer` **不能** 被直接导入.

实际上使用这种导入方法, 使用者也可以用一个完全不同的文件路径结构, 只要模块路径可以在 `PYTHONPATH` 里面被定位.

#### 3. 在配置文件中修改优化器

随后你可以在配置文件中的 `optimizer` 里使用 `MyOptimizer`. 如下所示, 在配置文件里, 优化器将会被字段 `optimizer` 来定义:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

如果要使用你自己的优化器, 字段可以被修改成:

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器构造器

一些模型的优化器可能会根据特定的参数而调整, 例如 BatchNorm 层的 weight decay. 使用者可以通过自定义优化器构造器来精细化设定不同参数的优化策略.

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

### 额外的设置

没有在优化器里被设置的训练技巧可以在优化器构造器 (例如逐个模型参数去设置学习率) 和钩子里实现. 我们列出了一些用来稳定或加速训练的常用设置, 如果想增加更多设置, 欢迎提交 issue 和 PR.

- __使用梯度裁剪 (gradient clip) 来稳定训练__:
  一些模型需要使用梯度裁剪来让训练过程更加稳定. 示例如下:

  ```python
  optim_wrapper = dict(
      _delete_=True,
      type='OptimWrapper',
      grad_clip=dict(max_norm=1, norm_type=2))
  ```

  如果你的配置文件继承自基配置文件, 后者已经设置了 `optim_wrapper`, 你需要使用 `_delete_=True` 来重写不必须的设置, 更多细节可以参考 [配置文件文档](https://mmsegmentation.readthedocs.io/en/latest/config.html).

## 自定义参数调度器

在自定义调度配置文件前, 推荐先了解 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md) 里面关于参数调度器的基本概念.

这里是一个参数调度器的例子:

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

**注意:** 当你修改 `train_cfg` 里面 `max_iters` 的时候, 请确保参数调度器 `param_scheduler` 里面的参数也被同时修改.
