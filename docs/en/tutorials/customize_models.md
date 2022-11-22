# Tutorial 4: Customize Models

## Customize optimizer

Assume you want to add a optimizer named as `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to first implement the new optimizer in a file, e.g., in `mmseg/core/optimizer/my_optimizer.py`:

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

Then add this module in `mmseg/core/optimizer/__init__.py` thus the registry will
find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed as

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `ADAM`, though the performance will drop a lot, the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

## Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNoarm layers.
The users can do those fine-grained parameter tuning through customizing optimizer constructor.

```
from mmcv.utils import build_from_cfg

from mmcv.runner import OPTIMIZER_BUILDERS
from .cocktail_optimizer import CocktailOptimizer


@OPTIMIZER_BUILDERS.register_module
class CocktailOptimizerConstructor(object):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

## Develop new components

There are mainly 2 types of components in MMSegmentation.

- backbone: usually stacks of convolutional network to extract feature maps, e.g., ResNet, HRNet.
- head: the component for semantic segmentation map decoding.

### Add new backbones

Here we show how to develop new components with an example of MobileNet.

1. Create a new file `mmseg/models/backbones/mobilenet.py`.

```python
import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

2. Import the module in `mmseg/models/backbones/__init__.py`.

```python
from .mobilenet import MobileNet
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add new heads

In MMSegmentation, we provide a base [BaseDecodeHead](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/decode_head.py) for all segmentation head.
All newly implemented decode heads should be derived from it.
Here we show how to develop a new head with the example of [PSPNet](https://arxiv.org/abs/1612.01105) as the following.

First, add a new decode head in `mmseg/models/decode_heads/psp_head.py`.
PSPNet implements a decode head for segmentation decode.
To implement a decode head, basically we need to implement three functions of the new module as the following.

```python
@HEADS.register_module()
class PSPHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)

    def init_weights(self):

    def forward(self, inputs):

```

Next, the users need to add the module in the `mmseg/models/decode_heads/__init__.py` thus the corresponding registry could find and load them.

To config file of PSPNet is as the following

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

### Add new loss

Assume you want to add a new loss as `MyLoss` for segmentation decode.
To add a new loss function, the users need implement it in `mmseg/models/losses/my_loss.py`.
The decorator `weighted_loss` enable the loss to be weighted for each element.

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

Then the users need to add it in the `mmseg/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss

```

To use it, modify the `loss_xxx` field.
Then you need to modify the `loss_decode` field in the head.
`loss_weight` could be used to balance multiple losses.

```python
loss_decode=dict(type='MyLoss', loss_weight=1.0))
```
