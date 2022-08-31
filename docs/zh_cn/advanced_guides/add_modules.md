# 自定义模型（待更新）

## 自定义优化器 (optimizer)

假设您想增加一个新的叫 `MyOptimizer` 的优化器，它的参数分别为 `a`, `b`, 和 `c`。
您首先需要在一个文件里实现这个新的优化器，例如在 `mmseg/core/optimizer/my_optimizer.py` 里面：

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

然后增加这个模块到 `mmseg/core/optimizer/__init__.py` 里面，这样注册器 (registry) 将会发现这个新的模块并添加它：

```python
from .my_optimizer import MyOptimizer
```

之后您可以在配置文件的 `optimizer` 域里使用 `MyOptimizer`，
如下所示，在配置文件里，优化器被 `optimizer` 域所定义：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

为了使用您自己的优化器，域可以被修改为：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

我们已经支持了 PyTorch 自带的全部优化器，唯一修改的地方是在配置文件里的 `optimizer` 域。例如，如果您想使用 `ADAM`，尽管数值表现会掉点，还是可以如下修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

使用者可以直接按照 PyTorch [文档教程](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 去设置参数。

## 定制优化器的构造器 (optimizer constructor)

对于优化，一些模型可能会有一些特别定义的参数，例如批归一化 (BatchNorm) 层里面的权重衰减 (weight decay)。
使用者可以通过定制优化器的构造器来微调这些细粒度的优化器参数。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner import OPTIMIZER_BUILDERS
from .cocktail_optimizer import CocktailOptimizer


@OPTIMIZER_BUILDERS.register_module
class CocktailOptimizerConstructor(object):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

## 开发和增加新的组件（Module）

MMSegmentation 里主要有2种组件：

- 主干网络 (backbone): 通常是卷积网络的堆叠，来做特征提取，例如 ResNet, HRNet
- 解码头 (decoder head): 用于语义分割图的解码的组件（得到分割结果）

### 添加新的主干网络

这里我们以 MobileNet 为例，展示如何增加新的主干组件：

1. 创建一个新的文件 `mmseg/models/backbones/mobilenet.py`

```python
import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

2. 在 `mmseg/models/backbones/__init__.py` 里面导入模块

```python
from .mobilenet import MobileNet
```

3. 在您的配置文件里使用它

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 增加新的解码头 (decoder head)组件

在 MMSegmentation 里面，对于所有的分割头，我们提供一个基类解码头 [BaseDecodeHead](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/decode_head.py) 。
所有新建的解码头都应该继承它。这里我们以 [PSPNet](https://arxiv.org/abs/1612.01105) 为例，
展示如何开发和增加一个新的解码头组件：

首先，在 `mmseg/models/decode_heads/psp_head.py` 里添加一个新的解码头。
PSPNet 中实现了一个语义分割的解码头。为了实现一个解码头，我们只需要在新构造的解码头中实现如下的3个函数：

```python
@HEADS.register_module()
class PSPHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)

    def init_weights(self):

    def forward(self, inputs):

```

接着，使用者需要在 `mmseg/models/decode_heads/__init__.py` 里面添加这个模块，这样对应的注册器 (registry) 可以查找并加载它们。

PSPNet的配置文件如下所示：

```python
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain_model/resnet50_v1c_trick-2cccc1ad.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

```

### 增加新的损失函数

假设您想添加一个新的损失函数 `MyLoss` 到语义分割解码器里。
为了添加一个新的损失函数，使用者需要在 `mmseg/models/losses/my_loss.py` 里面去实现它。
`weighted_loss` 可以对计算损失时的每个样本做加权。

```python
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
```

然后使用者需要在 `mmseg/models/losses/__init__.py` 里面添加它：

```python
from .my_loss import MyLoss, my_loss

```

为了使用它，修改 `loss_xxx` 域。之后您需要在解码头组件里修改 `loss_decode` 域。
`loss_weight` 可以被用来对不同的损失函数做加权。

```python
loss_decode=dict(type='MyLoss', loss_weight=1.0))
```
