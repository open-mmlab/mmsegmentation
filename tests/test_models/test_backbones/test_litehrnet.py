import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.backbones import LiteHRNet
from mmseg.models.backbones.litehrnet import LiteHRModule


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (_BatchNorm, )):
        return True
    return False


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def test_litehrmodule():
    # Test LiteHRModule forward
    block = LiteHRModule(
        num_branches=1,
        num_blocks=1,
        in_channels=[
            40,
        ],
        reduce_ratio=8,
        module_type='LITE')

    x = torch.randn(2, 40, 56, 56)
    x_out = block([[x]])
    assert x_out[0][0].shape == torch.Size([2, 40, 56, 56])

    block = LiteHRModule(
        num_branches=1,
        num_blocks=1,
        in_channels=[
            40,
        ],
        reduce_ratio=8,
        module_type='NAIVE')

    x = torch.randn(2, 40, 56, 56)
    x_out = block([x])
    assert x_out[0].shape == torch.Size([2, 40, 56, 56])

    with pytest.raises(ValueError):
        block = LiteHRModule(
            num_branches=1,
            num_blocks=1,
            in_channels=[
                40,
            ],
            reduce_ratio=8,
            module_type='none')


def test_litehrnet_backbone():
    # test LITE module
    extra = dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        with_head=True)

    model = LiteHRNet(extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 40, 56, 56])

    # test NAIVE module
    extra = dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        with_head=False)

    model = LiteHRNet(extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 40, 56, 56])

    # with_head=False
    extra = dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        with_head=False)

    model = LiteHRNet(extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 40, 56, 56])
    assert feat[1].shape == torch.Size([2, 80, 28, 28])
    assert feat[2].shape == torch.Size([2, 160, 14, 14])
    assert feat[3].shape == torch.Size([2, 320, 7, 7])

    # with_head=True
    extra = dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        with_head=True)

    model = LiteHRNet(extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 40, 56, 56])
    assert feat[1].shape == torch.Size([2, 40, 28, 28])
    assert feat[2].shape == torch.Size([2, 80, 14, 14])
    assert feat[3].shape == torch.Size([2, 160, 7, 7])
