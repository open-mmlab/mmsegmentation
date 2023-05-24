# Add New Modules

## Develop new components

We can customize all the components introduced at [the model documentation](./models.md), such as **backbone**, **head**, **loss function** and **data preprocessor**.

### Add new backbones

Here we show how to develop a new backbone with an example of MobileNet.

1. Create a new file `mmseg/models/backbones/mobilenet.py`.

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

In MMSegmentation, we provide a [BaseDecodeHead](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/decode_head.py#L17) for developing all segmentation heads.
All newly implemented decode heads should be derived from it.
Here we show how to develop a new head with the example of [PSPNet](https://arxiv.org/abs/1612.01105) as the following.

First, add a new decode head in `mmseg/models/decode_heads/psp_head.py`.
PSPNet implements a decode head for segmentation decode.
To implement a decode head, we need to implement three functions of the new module as the following.

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

Next, the users need to add the module in the `mmseg/models/decode_heads/__init__.py`, thus the corresponding registry could find and load them.

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
To add a new loss function, the users need to implement it in `mmseg/models/losses/my_loss.py`.
The decorator `weighted_loss` enables the loss to be weighted for each element.

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

### Add new data preprocessor

In MMSegmentation 1.x versions, we use [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/data_preprocessor.py#L13) to copy data to the target device and preprocess the data into the model input format as default. Here we show how to develop a new data preprocessor.

1. Create a new file `mmseg/models/my_datapreprocessor.py`.

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

2. Import your data preprocessor in `mmseg/models/__init__.py`

   ```python
   from .my_datapreprocessor import MyDataPreProcessor
   ```

3. Use it in your config file.

   ```python
   model = dict(
       data_preprocessor=dict(type='MyDataPreProcessor)
       ...
   )
   ```

## Develop new segmentors

The segmentor is an algorithmic architecture in which users can customize their algorithms by adding customized components and defining the logic of algorithm execution. Please refer to [the model document](./models.md) for more details.

Since the [BaseSegmentor](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/segmentors/base.py#L15) in MMSegmentation unifies three modes for a forward process, to develop a new segmentor, users need to overwrite `loss`, `predict` and `_forward` methods corresponding to the `loss`, `predict` and `tensor` modes.

Here we show how to develop a new segmentor.

1. Create a new file `mmseg/models/segmentors/my_segmentor.py`.

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

2. Import your segmentor in `mmseg/models/segmentors/__init__.py`.

   ```python
   from .my_segmentor import MySegmentor
   ```

3. Use it in your config file.

   ```python
   model = dict(
       type='MySegmentor'
       ...
   )
   ```
