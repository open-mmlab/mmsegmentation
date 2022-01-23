import pytest
import torch

from mmseg.models.backbones.efficientnet import EfficientNet


def test_efficientnet():
    # Test normal input
    H, W = (224, 224)
    temp = torch.randn((10, 3, H, W))
    model = EfficientNet(
        strides=(1, 2, 2, 2, 1, 2, 1),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='Swish'),
        with_cp=False,
        pretrained=None,
        model_name='efficientnet-b0',
        init_cfg=None
    )
    model.init_weights()
    outs = model(temp)
    assert outs.shape == (10, 1280, 1, 1)


def test_EfficientNet_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = EfficientNet(pretrained=None, init_cfg=None)
    assert model.init_cfg == [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = EfficientNet(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = EfficientNet(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = EfficientNet(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = EfficientNet(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = EfficientNet(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = EfficientNet(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = EfficientNet(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = EfficientNet(pretrained=123, init_cfg=123)