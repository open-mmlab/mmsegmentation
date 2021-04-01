import pytest
import torch

from mmseg.models.backbones import CGNet
from mmseg.models.backbones.cgnet import (ContextGuidedBlock,
                                          GlobalContextExtractor)


def test_cgnet_GlobalContextExtractor():
    block = GlobalContextExtractor(16, 16, with_cp=True)
    x = torch.randn(2, 16, 64, 64, requires_grad=True)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 16, 64, 64])


def test_cgnet_context_guided_block():
    with pytest.raises(AssertionError):
        # cgnet ContextGuidedBlock GlobalContextExtractor channel and reduction
        # constraints.
        ContextGuidedBlock(8, 8)

    # test cgnet ContextGuidedBlock with checkpoint forward
    block = ContextGuidedBlock(
        16, 16, act_cfg=dict(type='PReLU'), with_cp=True)
    assert block.with_cp
    x = torch.randn(2, 16, 64, 64, requires_grad=True)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 16, 64, 64])

    # test cgnet ContextGuidedBlock without checkpoint forward
    block = ContextGuidedBlock(32, 32)
    assert not block.with_cp
    x = torch.randn(3, 32, 32, 32)
    x_out = block(x)
    assert x_out.shape == torch.Size([3, 32, 32, 32])

    # test cgnet ContextGuidedBlock with down sampling
    block = ContextGuidedBlock(32, 32, downsample=True)
    assert block.conv1x1.conv.in_channels == 32
    assert block.conv1x1.conv.out_channels == 32
    assert block.conv1x1.conv.kernel_size == (3, 3)
    assert block.conv1x1.conv.stride == (2, 2)
    assert block.conv1x1.conv.padding == (1, 1)

    assert block.f_loc.in_channels == 32
    assert block.f_loc.out_channels == 32
    assert block.f_loc.kernel_size == (3, 3)
    assert block.f_loc.stride == (1, 1)
    assert block.f_loc.padding == (1, 1)
    assert block.f_loc.groups == 32
    assert block.f_loc.dilation == (1, 1)
    assert block.f_loc.bias is None

    assert block.f_sur.in_channels == 32
    assert block.f_sur.out_channels == 32
    assert block.f_sur.kernel_size == (3, 3)
    assert block.f_sur.stride == (1, 1)
    assert block.f_sur.padding == (2, 2)
    assert block.f_sur.groups == 32
    assert block.f_sur.dilation == (2, 2)
    assert block.f_sur.bias is None

    assert block.bottleneck.in_channels == 64
    assert block.bottleneck.out_channels == 32
    assert block.bottleneck.kernel_size == (1, 1)
    assert block.bottleneck.stride == (1, 1)
    assert block.bottleneck.bias is None

    x = torch.randn(1, 32, 32, 32)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 32, 16, 16])

    # test cgnet ContextGuidedBlock without down sampling
    block = ContextGuidedBlock(32, 32, downsample=False)
    assert block.conv1x1.conv.in_channels == 32
    assert block.conv1x1.conv.out_channels == 16
    assert block.conv1x1.conv.kernel_size == (1, 1)
    assert block.conv1x1.conv.stride == (1, 1)
    assert block.conv1x1.conv.padding == (0, 0)

    assert block.f_loc.in_channels == 16
    assert block.f_loc.out_channels == 16
    assert block.f_loc.kernel_size == (3, 3)
    assert block.f_loc.stride == (1, 1)
    assert block.f_loc.padding == (1, 1)
    assert block.f_loc.groups == 16
    assert block.f_loc.dilation == (1, 1)
    assert block.f_loc.bias is None

    assert block.f_sur.in_channels == 16
    assert block.f_sur.out_channels == 16
    assert block.f_sur.kernel_size == (3, 3)
    assert block.f_sur.stride == (1, 1)
    assert block.f_sur.padding == (2, 2)
    assert block.f_sur.groups == 16
    assert block.f_sur.dilation == (2, 2)
    assert block.f_sur.bias is None

    x = torch.randn(1, 32, 32, 32)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 32, 32, 32])


def test_cgnet_backbone():
    with pytest.raises(AssertionError):
        # check invalid num_channels
        CGNet(num_channels=(32, 64, 128, 256))

    with pytest.raises(AssertionError):
        # check invalid num_blocks
        CGNet(num_blocks=(3, 21, 3))

    with pytest.raises(AssertionError):
        # check invalid dilation
        CGNet(num_blocks=2)

    with pytest.raises(AssertionError):
        # check invalid reduction
        CGNet(reductions=16)

    with pytest.raises(AssertionError):
        # check invalid num_channels and reduction
        CGNet(num_channels=(32, 64, 128), reductions=(64, 129))

    # Test CGNet with default settings
    model = CGNet()
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([2, 35, 112, 112])
    assert feat[1].shape == torch.Size([2, 131, 56, 56])
    assert feat[2].shape == torch.Size([2, 256, 28, 28])

    # Test CGNet with norm_eval True and with_cp True
    model = CGNet(norm_eval=True, with_cp=True)
    with pytest.raises(TypeError):
        # check invalid pretrained
        model.init_weights(pretrained=8)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([2, 35, 112, 112])
    assert feat[1].shape == torch.Size([2, 131, 56, 56])
    assert feat[2].shape == torch.Size([2, 256, 28, 28])
