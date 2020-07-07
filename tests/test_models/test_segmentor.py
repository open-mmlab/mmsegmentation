import mmcv
import numpy as np
import torch
from torch import nn

from mmseg.models import BACKBONES, HEADS, build_segmentor
from mmseg.models.decode_heads.cascade_decode_head import BaseCascadeDecodeHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


def _demo_mm_inputs(input_shape=(1, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


@BACKBONES.register_module()
class ExampleBackbone(nn.Module):

    def __init__(self):
        super(ExampleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


@HEADS.register_module()
class ExampleDecodeHead(BaseDecodeHead):

    def __init__(self):
        super(ExampleDecodeHead, self).__init__(3, 3, num_classes=19)

    def forward(self, inputs):
        return self.cls_seg(inputs[0])


@HEADS.register_module()
class ExampleCascadeDecodeHead(BaseCascadeDecodeHead):

    def __init__(self):
        super(ExampleCascadeDecodeHead, self).__init__(3, 3, num_classes=19)

    def forward(self, inputs, prev_out):
        return self.cls_seg(inputs[0])


def _segmentor_forward_train_test(segmentor):
    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs(num_classes=num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_semantic_seg = mm_inputs['gt_semantic_seg']

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        segmentor = segmentor.cuda()
        imgs = imgs.cuda()
        gt_semantic_seg = gt_semantic_seg.cuda()

    # Test forward train
    losses = segmentor.forward(
        imgs, img_metas, gt_semantic_seg=gt_semantic_seg, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward simple test
    with torch.no_grad():
        segmentor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_meta_list = [[img_meta] for img_meta in img_metas]
        segmentor.forward(img_list, img_meta_list, return_loss=False)

    # Test forward aug test
    with torch.no_grad():
        segmentor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_list = img_list + img_list
        img_meta_list = [[img_meta] for img_meta in img_metas]
        img_meta_list = img_meta_list + img_meta_list
        segmentor.forward(img_list, img_meta_list, return_loss=False)


def test_encoder_decoder():

    # test 1 decode head, w.o. aux head
    cfg = dict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'))
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test slide mode
    test_cfg = mmcv.Config(dict(mode='slide', crop_size=(3, 3), stride=(2, 2)))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 1 aux head
    cfg = dict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=dict(type='ExampleDecodeHead'))
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 2 aux head
    cfg = dict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleDecodeHead')
        ])
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)


def test_cascade_encoder_decoder():

    # test 1 decode head, w.o. aux head
    cfg = dict(
        type='CascadeEncoderDecoder',
        num_stages=2,
        backbone=dict(type='ExampleBackbone'),
        decode_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleCascadeDecodeHead')
        ])
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test slide mode
    test_cfg = mmcv.Config(dict(mode='slide', crop_size=(3, 3), stride=(2, 2)))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 1 aux head
    cfg = dict(
        type='CascadeEncoderDecoder',
        num_stages=2,
        backbone=dict(type='ExampleBackbone'),
        decode_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleCascadeDecodeHead')
        ],
        auxiliary_head=dict(type='ExampleDecodeHead'))
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 2 aux head
    cfg = dict(
        type='CascadeEncoderDecoder',
        num_stages=2,
        backbone=dict(type='ExampleBackbone'),
        decode_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleCascadeDecodeHead')
        ],
        auxiliary_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleDecodeHead')
        ])
    test_cfg = mmcv.Config(dict(mode='whole'))
    segmentor = build_segmentor(cfg, train_cfg=None, test_cfg=test_cfg)
    _segmentor_forward_train_test(segmentor)
