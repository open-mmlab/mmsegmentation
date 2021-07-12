# 教程 6: 自定义运行设定

## 自定义优化设定

### 自定义 PyTorch 支持的优化器（optimizer）

我们已经支持 PyTorch 自带的所有优化器，唯一需要修改的地方是改变配置文件中的 `optimizer` 字段。例如，如果您想使用 `ADAM`（注意如下操作可能会让模型表现下降），可以使用如下修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

要修改模型的学习率，使用者只需要修改配置文件里 `optimizer` 中的 `lr` 即可。使用者可以参照 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)直接设置参数。

### 定制自己实现的优化器（optimizer）

#### 1. 定义一个新的优化器（optimizer）

一个自定义的优化器可以按照如下去定义：

假如您想增加一个名为 `MyOptimizer` 的优化器，它有参数 `a`, `b`, 和 `c`。您需要创建一个名为 `mmseg/core/optimizer` 的新文件夹。然后在一个文件中实现新的优化器，例如，在  `mmseg/core/optimizer/my_optimizer.py` 中实现一个新优化器：

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)为

```

#### 2. 将优化器添加到注册表（registry）中

为了让上述定义的模块被框架找到，首先这个模块应该被导入到主命名空间（main namespace）里。有两种方式可以实现它：

- 修改 `mmseg/core/optimizer/__init__.py` 来导入它。

    新定义的模块应该被导入到 `mmseg/core/optimizer/__init__.py` ，这样注册表将会找到新模块并添加它。

    ```python
    from .my_optimizer import MyOptimizer
    ```

- 在配置文件里使用 `custom_imports` 来手动导入它。

  ```python
  custom_imports = dict(imports=['mmseg.core.optimizer.my_optimizer'], allow_failed_imports=False)
  ```

`mmseg.core.optimizer.my_optimizer` 模块将会在程序开始运行时被导入，然后 `MyOptimizer` 类将会被自动注册。注意，只有包含 `MyOptimizer`  类的包（package）应该被导入。 `mmseg.core.optimizer.my_optimizer.MyOptimizer` **不能** 被直接导入。

事实上，使用者完全可以用这种导入方法使用完全不同的文件目录结构，只要模块的根路径已经被添加到 `PYTHONPATH` 里面。

#### 3. 在配置文件里指定优化器（optimizer）

之后您可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer` 。在配置文件里，优化器被定义在 `optimizer` 字段里，如下所示：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

为了使用您自己的优化器，这个字段可以被改成：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器的构造器（constructor）

一些模型可能有一些特定的优化参数设置，例如批归一化（BatchNorm）层里面的权重衰减 （weight decay）。使用者可以通过定制优化器的构造器来进行这些细粒度的参数调整。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmseg.utils import get_root_logger
from .my_optimizer import MyOptimizer


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(object):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

默认的优化器构造器的实现可以参照 [这里](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11) ，它也可以作为新的优化器构造器的模板。

### 额外的设置

优化器没有实现的技巧应该通过优化器构造器（optimizer constructor）或者钩子（hook）去实现，如设置基于参数的学习率（parameter-wise learning rates）。我们列出一些可以稳定或加速模型的训练的常用设置。如果您有更多的设置，欢迎提交 PR 和创建 issue 。

- __使用梯度截断（gradient clip）去稳定训练__：
    一些模型需要用梯度截断去稳定训练过程，如下所示：

    ```python
    optimizer_config = dict(
        _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    ```

    如果您的配置继承自已经设置了  `optimizer_config` 的基础配置（base config），您可能需要 `_delete_=True` 来覆盖不需要的设置。更多细节请参照 [配置文件文档](https://mmsegmentation.readthedocs.io/en/latest/config.html) 。

- __使用动量策略（momentum schedule）去加速模型收敛__:
    我们支持动量策略去让模型基于学习率修改动量，这样可以让模型收敛地更快。
    动量策略经常和学习率计划表（LR scheduler）一起使用，例如，在 3D 检测中经常使用一下配置来加速收敛。更多细节请参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130) 的实现。

    ```python
    lr_config = dict(
        policy='cyclic',
        target_ratio=(10, 1e-4),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    momentum_config = dict(
        policy='cyclic',
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    ```

## 自定义训练策略

默认情况下，我们根据训练迭代步数 40k/80k 设置学习率，这在 MMCV 里叫做 [`PolyLrUpdaterHook`](https://github.com/open-mmlab/mmcv/blob/826d3a7b68596c824fa1e2cb89b6ac274f52179c/mmcv/runner/hooks/lr_updater.py#L196) 。我们也支持许多其他的学习率策略，如 [这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py) 所示，例如 `CosineAnnealing` 和 `Poly` 计划表。下面是一些例子：

- 等间隔调整策略 Step schedule：

    ```python
    lr_config = dict(policy='step', step=[9, 10])
    ```

- 余弦退火策略 ConsineAnnealing schedule：

    ```python
    lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1.0 / 10,
        min_lr_ratio=1e-5)
    ```

## 自定义工作流 (workflow)

工作流是一个形式为 (阶段, 历时) 的列表，用于指定运行顺序和轮数。默认情况下，它被设置成：

```python
workflow = [('train', 1)]
```

意思是运行 1 个 epoch来进行训练。有时，使用者可能想在验证集上检查一些关于模型的指标（如 损失 loss，精确性 accuracy）。这种情况下，我们可以将工作流设置为：

```python
[('train', 1), ('val', 1)]
```

这样一来，1 个 epoch 的训练，1 个 epoch 的验证将被反复运行。

**注意**:

1. 模型的参数在验证的阶段不会被更新。
2. 配置文件里的关键词 `total_epochs` 仅控制训练的 epochs 数目，而不会影响验证时的工作流。
3. 工作流 `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 不会改变 `EvalHook` 的行为，因为 `EvalHook` 是由 `after_train_epoch` 调用的，验证的工作流仅仅影响通过 `after_val_epoch` 调用的钩子（hooks）。因此， `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 的唯一区别是，runner 将在每次训练 epoch 结束后计算在验证集上的损失。

## 自定义钩子（hooks）

### 使用在 MMCV 中实现的钩子（hooks）

如果钩子已经在 MMCV 里实现，您可以直接修改配置文件来使用钩子，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### 修改默认的运行时钩子（runtime hooks）

以下的常用的钩子没有通过 `custom_hooks` 注册：

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

在这些钩子中，只有 logger hook 的优先级为 `VERY_LOW` ，其他的优先级都是 `NORMAL`。上述教程已经涵盖了如何修改 `optimizer_config`，`momentum_config` 和 `lr_config`。
这里我们展示我们如何对 `log_config`， `checkpoint_config` 和 `evaluation` 进行操作。

#### 检查点配置文件（Checkpoint config）

MMCV runner 将使用 `checkpoint_config` 去初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9).

```python
checkpoint_config = dict(interval=1)
```

使用者可以设置 `max_keep_ckpts` 来只保存少量的检查点，或者通过 `save_optimizer` 来决定是否保存优化器的状态字典 （state dict of optimizer）。 参数的更多细节请参考 [这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook) 。

#### 日志配置文件 （Log config）

`log_config` 包裹了多个日志钩子（logger hooks）而且可以设置间隔（intervals）。现在 MMCV 支持 `WandbLoggerHook`， `MlflowLoggerHook` 和 `TensorboardLoggerHook`。详细的使用方法请参照 [文档](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook) 。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 评估配置文件（Evaluation config）

`evaluation` 的配置文件将被用来初始化 [`EvalHook`](https://github.com/open-mmlab/mmsegmentation/blob/e3f6f655d69b777341aec2fe8829871cc0beadcb/mmseg/core/evaluation/eval_hooks.py#L7) 。除了关键的 `interval` ，其他的参数如 `metric` 将被传递给 `dataset.evaluate()` 。

```python
evaluation = dict(interval=1, metric='mIoU')
```
