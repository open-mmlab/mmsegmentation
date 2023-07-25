# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.optim import OptimWrapper
from mmengine.structures import PixelData
from torch import nn
from torch.optim import SGD

from mmseg.models import SegDataPreProcessor
from mmseg.models.decode_heads.cascade_decode_head import BaseCascadeDecodeHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


def _demo_mm_inputs(input_shape=(1, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    imgs = torch.randn(*input_shape)
    segs = torch.randint(
        low=0, high=num_classes - 1, size=(N, H, W), dtype=torch.long)

    img_metas = [{
        'img_shape': (H, W),
        'ori_shape': (H, W),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    data_samples = [
        SegDataSample(
            gt_sem_seg=PixelData(data=segs[i]), metainfo=img_metas[i])
        for i in range(N)
    ]

    mm_inputs = {'imgs': torch.FloatTensor(imgs), 'data_samples': data_samples}

    return mm_inputs


@MODELS.register_module()
class ExampleBackbone(nn.Module):

    def __init__(self, out_indices=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.out_indices = out_indices

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        if self.out_indices is None:
            return [self.conv(x)]
        else:
            outs = []
            for i in self.out_indices:
                outs.append(self.conv(x))
        return outs


@MODELS.register_module()
class ExampleDecodeHead(BaseDecodeHead):

    def __init__(self, num_classes=19, out_channels=None, **kwargs):
        super().__init__(
            3, 3, num_classes=num_classes, out_channels=out_channels, **kwargs)

    def forward(self, inputs):
        return self.cls_seg(inputs[0])


@MODELS.register_module()
class ExampleTextEncoder(nn.Module):

    def __init__(self, vocabulary=None, output_dims=None):
        super().__init__()
        self.vocabulary = vocabulary
        self.output_dims = output_dims

    def forward(self):
        return torch.randn((len(self.vocabulary), self.output_dims))


@MODELS.register_module()
class ExampleCascadeDecodeHead(BaseCascadeDecodeHead):

    def __init__(self):
        super().__init__(3, 3, num_classes=19)

    def forward(self, inputs, prev_out):
        return self.cls_seg(inputs[0])


def _segmentor_forward_train_test(segmentor):
    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs(num_classes=num_classes)

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        segmentor = segmentor.cuda()

    # check data preprocessor
    if not hasattr(segmentor,
                   'data_preprocessor') or segmentor.data_preprocessor is None:
        segmentor.data_preprocessor = SegDataPreProcessor()

    mm_inputs = segmentor.data_preprocessor(mm_inputs, True)
    imgs = mm_inputs.pop('imgs')
    data_samples = mm_inputs.pop('data_samples')

    # create optimizer wrapper
    optimizer = SGD(segmentor.parameters(), lr=0.1)
    optim_wrapper = OptimWrapper(optimizer)

    # Test forward train
    losses = segmentor.forward(imgs, data_samples, mode='loss')
    assert isinstance(losses, dict)

    # Test train_step
    data_batch = dict(inputs=imgs, data_samples=data_samples)
    outputs = segmentor.train_step(data_batch, optim_wrapper)
    assert isinstance(outputs, dict)
    assert 'loss' in outputs

    # Test val_step
    with torch.no_grad():
        segmentor.eval()
        data_batch = dict(inputs=imgs, data_samples=data_samples)
        outputs = segmentor.val_step(data_batch)
        assert isinstance(outputs, list)

    # Test forward simple test
    with torch.no_grad():
        segmentor.eval()
        data_batch = dict(inputs=imgs, data_samples=data_samples)
        results = segmentor.forward(imgs, data_samples, mode='tensor')
        assert isinstance(results, torch.Tensor)


def _segmentor_predict(segmentor):
    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs(num_classes=num_classes)

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        segmentor = segmentor.cuda()

    # check data preprocessor
    if not hasattr(segmentor,
                   'data_preprocessor') or segmentor.data_preprocessor is None:
        segmentor.data_preprocessor = SegDataPreProcessor()

    mm_inputs = segmentor.data_preprocessor(mm_inputs, True)
    imgs = mm_inputs.pop('imgs')
    data_samples = mm_inputs.pop('data_samples')

    # Test predict
    with torch.no_grad():
        segmentor.eval()
        data_batch = dict(inputs=imgs, data_samples=data_samples)
        outputs = segmentor.predict(**data_batch)
        assert isinstance(outputs, list)
