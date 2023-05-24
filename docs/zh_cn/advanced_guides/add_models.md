# 新增模块

## 开发新组件

我们可以自定义 [模型文档](./models.md) 中介绍的所有组件，例如**主干网络（backbone）**、**头（head）**、**损失函数（loss function）**和**数据预处理器（data preprocessor）**。

### 添加新的主干网络（backbone）

在这里，我们以 MobileNet 为例展示如何开发新的主干网络。

1. 创建一个新文件 `mmseg/models/backbones/mobilenet.py`。

   ```python
   import torch.nn as nn

   from mmseg.registry import MODELS


   @MODELS.register_module()
   class MobileNet(nn.Module):

       def __init__(self, arg1, arg2):
           pass

       def forward(self, x):  # should return a tuple
           pass

       def init_weights(self, pretrained=None):
           pass
   ```

2. 在 `mmseg/models/backbones/__init__.py` 中引入模块。

   ```python
   from .mobilenet import MobileNet
   ```

3. 在配置文件中使用它。

   ```python
   model = dict(
       ...
       backbone=dict(
           type='MobileNet',
           arg1=xxx,
           arg2=xxx),
       ...
   ```

### 添加新的头（head）

在 MMSegmentation 中，我们提供 [BaseDecodeHead](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/decode_heads/decode_head.py#L17) 用于开发所有分割头。
所有新实现的解码头都应该从中派生出来。
接下来我们以 [PSPNet](https://arxiv.org/abs/1612.01105) 为例说明如何开发新的头。

首先，在 `mmseg/models/decode_heads/psp_head.py` 中添加一个新的解码头。
PSPNet 实现了用于分割解码的解码头。
为了实现解码头，在新模块中我们需要执行以下三个函数。

```python
from mmseg.registry import MODELS

@MODELS.register_module()
class PSPHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)

    def init_weights(self):
        pass

    def forward(self, inputs):
        pass
```

接下来，用户需要在 `mmseg/models/decode_heads/__init__.py` 中添加模块，这样相应的注册器就可以找到并加载它们。

PSPNet 的配置文件如下

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

### 添加新的损失函数（loss）

假设您想为分割解码添加一个叫做 `MyLoss` 的新的损失函数。
要添加新的损失函数，用户需要在 `mmseg/models/loss/my_loss.py` 中实现它。
修饰器 `weighted_loss` 可以对损失的每个元素进行加权。

```python
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@MODELS.register_module()
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

然后，用户需要将其添加到 `mmseg/models/loss/__init__.py` 中。

```python
from .my_loss import MyLoss, my_loss

```

要使用它，请修改 `loss_xx` 字段。
然后需要修改头中的 `loss_decode` 字段。
`loss_weight` 可用于平衡多重损失。

```python
loss_decode=dict(type='MyLoss', loss_weight=1.0))
```

### 添加新的数据预处理器（data preprocessor）

在 MMSegmentation 1.x 版本中，我们使用 [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/data_preprocessor.py#L13) 将数据复制到目标设备，并将数据预处理为默认的模型输入格式。这里我们将展示如何开发一个新的数据预处理器。

1. 创建一个新文件 `mmseg/models/my_datapreprocessor.py`。

   ```python
   from mmengine.model import BaseDataPreprocessor

   from mmseg.registry import MODELS

   @MODELS.register_module()
   class MyDataPreProcessor(BaseDataPreprocessor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)

       def forward(self, data: dict, training: bool=False) -> Dict[str, Any]:
           # TODO Define the logic for data pre-processing in the forward method
           pass
   ```

2. 在 `mmseg/models/__init__.py` 中导入数据预处理器

   ```python
   from .my_datapreprocessor import MyDataPreProcessor
   ```

3. 在配置文件中使用它。

   ```python
   model = dict(
       data_preprocessor=dict(type='MyDataPreProcessor)
       ...
   )
   ```

## 开发新的分割器（segmentor）

分割器是一种户可以通过添加自定义组件和定义算法执行逻辑来自定义其算法的算法架构。请参考[模型文档](./models.md)了解更多详情。

由于 MMSegmentation 中的 [BaseSegmenter](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/segmentors/base.py#L15) 统一了前向过程的三种模式，为了开发新的分割器，用户需要重写与 `loss`、`predict` 和 `tensor` 相对应的 `loss`、`predict` 和 `_forward` 方法。

这里我们将展示如何开发一个新的分割器。

1. 创建一个新文件 `mmseg/models/segmentors/my_segmentor.py`。

   ```python
    from typing import Dict, Optional, Union

    import torch

    from mmseg.registry import MODELS
    from mmseg.models import BaseSegmentor

    @MODELS.register_module()
    class MySegmentor(BaseSegmentor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # TODO users should build components of the network here

        def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
            """Calculate losses from a batch of inputs and data samples."""
            pass

        def predict(self, inputs: Tensor, data_samples: OptSampleList=None) -> SampleList:
            """Predict results from a batch of inputs and data samples with post-
            processing."""
            pass

       def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
            """Network forward process.

            Usually includes backbone, neck and head forward without any post-
            processing.
            """
            pass
   ```

2. 在 `mmseg/models/segmentors/__init__.py` 中导入分割器。

   ```python
   from .my_segmentor import MySegmentor
   ```

3. 在配置文件中使用它。

   ```python
   model = dict(
       type='MySegmentor'
       ...
   )
   ```
