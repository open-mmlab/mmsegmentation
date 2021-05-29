import pytest
import torch

from mmseg.models.backbones import ResNeXt
from mmseg.models.backbones.resnext import Bottleneck as BottleneckX
from .utils import is_block


def test_renext_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckX(64, 64, groups=32, base_width=4, style='tensorflow')

    # Test ResNeXt Bottleneck structure
    block = BottleneckX(
        64, 64, groups=32, base_width=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    # Test ResNeXt Bottleneck with DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        # conv_cfg must be None if dcn is not None
        BottleneckX(
            64,
            64,
            groups=32,
            base_width=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    BottleneckX(64, 64, dcn=dcn)

    # Test ResNeXt Bottleneck forward
    block = BottleneckX(64, 16, groups=32, base_width=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnext_backbone():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

    # Test ResNeXt with group 32, base_width 4
    model = ResNeXt(depth=50, groups=32, base_width=4)
    print(model)
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])
