import pytest
import torch

from mmseg.models.backbones.convnext import ConvNeXt


def test_convnext():
    # Test normal input
    H, W = (512, 512)
    temp = torch.randn((1, 3, H, W))
    dims = [96, 192, 384, 768]
    model = ConvNeXt(
        in_chans=3,
        num_stages=4,
        depths=[3, 3, 9, 3],
        dims=dims,
        kernel_size=7,
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'))
    model.init_weights()
    outs = model(temp)
    assert outs[0].shape == (1, dims[0], H // 4, W // 4)
    assert outs[1].shape == (1, dims[1], H // 8, W // 8)
    assert outs[2].shape == (1, dims[2], H // 16, W // 16)
    assert outs[3].shape == (1, dims[3], H // 32, W // 32)


def test_convnext_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = ConvNeXt(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = ConvNeXt(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = ConvNeXt(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = ConvNeXt(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = ConvNeXt(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = ConvNeXt(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = ConvNeXt(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = ConvNeXt(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = ConvNeXt(pretrained=123, init_cfg=123)
