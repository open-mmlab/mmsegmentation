import pytest
import torch

from mmseg.models.backbones.mobilenet_v2 import MobileNetV2


def test_mobile_net_init():
    # init_cfg and pretrain can't be given at the same time
    with pytest.raises(AssertionError):
        MobileNetV2(pretrained='test', init_cfg=dict(type='Pretrained'))
    # use pretrain
    net = MobileNetV2(pretrained='test')
    assert net.init_cfg == dict(
        type='Pretrained', checkpoint='test', prefix='backbone.')
    # use init_cfg
    net = MobileNetV2(init_cfg=dict(type='Pretrained', checkpoint='test'))
    assert net.init_cfg == dict(
        type='Pretrained', checkpoint='test', prefix='backbone.')

    # default init
    net = MobileNetV2()
    assert net.init_cfg == [
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
    ]
    # init_cfg type error
    with pytest.raises(TypeError):
        MobileNetV2(init_cfg='wrong')

    # pretrain type error
    with pytest.raises(TypeError):
        MobileNetV2(pretrained=123)


def test_mobile_net():
    # test output index out of range
    with pytest.raises(ValueError):
        MobileNetV2(out_indices=(8, 7))
    # test frozen states out of range
    with pytest.raises(ValueError):
        MobileNetV2(frozen_stages=9)
    # test forward with single output
    imgs = torch.randn(2, 3, 56, 56)
    net = MobileNetV2(out_indices=(5, ))
    output = net(imgs)
    assert isinstance(output, torch.Tensor)
    # test frozen staes
    net = MobileNetV2(frozen_stages=2, norm_eval=True)
    net.train()
