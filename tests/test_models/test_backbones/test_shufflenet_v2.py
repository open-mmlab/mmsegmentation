import pytest
import torch
from mmcv.utils import is_tuple_of
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.backbones import ShuffleNetV2
from mmseg.models.backbones.shufflenet_v2 import InvertedResidual
from .utils import check_norm_state, is_norm


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (InvertedResidual, )):
        return True
    return False


def test_shufflenetv2_invertedresidual():
    # Test InvertedResidual forward, stride=1, dilation=1
    block = InvertedResidual(24, 24, stride=1)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test InvertedResidual forward, stride=1, dilation=2
    block = InvertedResidual(24, 24, stride=1, dilation=2)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test InvertedResidual forward, stride=1, dilation=4
    block = InvertedResidual(24, 24, stride=1, dilation=4)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test InvertedResidual forward, stride=2, dilation=1
    block = InvertedResidual(24, 48, stride=2)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 28, 28))

    # Test InvertedResidual forward, stride=2, dilation=2
    block = InvertedResidual(24, 48, stride=2, dilation=2)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 28, 28))

    # Test InvertedResidual forward, stride=2, dilation=4
    block = InvertedResidual(24, 48, stride=2, dilation=4)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 28, 28))

    # Test InvertedResidual with checkpoint forward
    block = InvertedResidual(48, 48, stride=1, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 48, 56, 56)
    x.requires_grad = True
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 56, 56))


def test_shufflenetv2_backbone():

    with pytest.raises(ValueError):
        # groups must be in 0.5, 1.0, 1.5, 2.0]
        ShuffleNetV2(widen_factor=3.0)

    with pytest.raises(ValueError):
        # frozen_stages must be in [0, 1, 2, 3, 4]
        ShuffleNetV2(widen_factor=1.0, frozen_stages=5)

    with pytest.raises(ValueError):
        # out_indices must be in [0, 1, 2, 3]
        ShuffleNetV2(widen_factor=1.0, out_indices=(4, ))

    with pytest.raises(TypeError):
        # pretrained must be str or None
        model = ShuffleNetV2()
        model.init_weights(pretrained=1)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations)
        ShuffleNetV2(
            stage_blocks=(4, 8, 4), strides=(2, 2), dilations=(1, 1, 1))

    # Test ShuffleNetV2 norm state
    model = ShuffleNetV2()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test ShuffleNetV2 with first stage frozen, frozen_stages=1
    frozen_stages = 1
    model = ShuffleNetV2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ShuffleNetV2 with first stage frozen, frozen_stages=4
    frozen_stages = 4
    model = ShuffleNetV2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ShuffleNetV2 with norm_eval
    model = ShuffleNetV2(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

    # Test ShuffleNetV2 forward with widen_factor=0.5
    model = ShuffleNetV2(widen_factor=0.5, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 48, 28, 28))
    assert feat[1].shape == torch.Size((1, 96, 14, 14))
    assert feat[2].shape == torch.Size((1, 192, 7, 7))
    assert feat[3].shape == torch.Size((1, 1024, 7, 7))

    # Test ShuffleNetV2 forward with widen_factor=1.0
    model = ShuffleNetV2(widen_factor=1.0, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 28, 28))
    assert feat[1].shape == torch.Size((1, 232, 14, 14))
    assert feat[2].shape == torch.Size((1, 464, 7, 7))
    assert feat[3].shape == torch.Size((1, 1024, 7, 7))

    # Test ShuffleNetV2 forward with widen_factor=1.5
    model = ShuffleNetV2(widen_factor=1.5, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 176, 28, 28))
    assert feat[1].shape == torch.Size((1, 352, 14, 14))
    assert feat[2].shape == torch.Size((1, 704, 7, 7))
    assert feat[3].shape == torch.Size((1, 1024, 7, 7))

    # Test ShuffleNetV2 forward with widen_factor=2.0
    model = ShuffleNetV2(widen_factor=2.0, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 244, 28, 28))
    assert feat[1].shape == torch.Size((1, 488, 14, 14))
    assert feat[2].shape == torch.Size((1, 976, 7, 7))
    assert feat[3].shape == torch.Size((1, 2048, 7, 7))

    # Test ShuffleNetV2 forward with layers 3 forward
    model = ShuffleNetV2(widen_factor=1.0, out_indices=(2, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 464, 7, 7))

    # Test ShuffleNetV2 forward with layers 1 2 forward
    model = ShuffleNetV2(widen_factor=1.0, out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 2
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 232, 14, 14))
    assert feat[1].shape == torch.Size((1, 464, 7, 7))

    # Test ShuffleNetV2 forward with checkpoint forward
    model = ShuffleNetV2(widen_factor=1.0, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

    # Test ShuffleNetV2 forward with widen_factor=1.0, stage_blocks=(4, 8, 4),
    # strides=(2, 1, 1), dilations=(1, 2, 4)
    model = ShuffleNetV2(
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3),
        strides=(2, 1, 1),
        dilations=(1, 2, 4))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 28, 28))
    assert feat[1].shape == torch.Size((1, 232, 28, 28))
    assert feat[2].shape == torch.Size((1, 464, 28, 28))
    assert feat[3].shape == torch.Size((1, 1024, 28, 28))
    assert model.layers[0][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[1][0].branch2[1].conv.dilation == (2, 2)
    assert model.layers[2][0].branch2[1].conv.dilation == (4, 4)

    # Test ShuffleNetV2 forward with widen_factor=1.0, stage_blocks=(4, 8, 4),
    # strides=(1, 1, 1), dilations=(2, 2, 4)
    model = ShuffleNetV2(
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3),
        strides=(1, 1, 1),
        dilations=(2, 2, 4))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 56, 56))
    assert feat[1].shape == torch.Size((1, 232, 56, 56))
    assert feat[2].shape == torch.Size((1, 464, 56, 56))
    assert feat[3].shape == torch.Size((1, 1024, 56, 56))
    assert model.layers[0][0].branch2[1].conv.dilation == (2, 2)
    assert model.layers[1][0].branch2[1].conv.dilation == (2, 2)
    assert model.layers[2][0].branch2[1].conv.dilation == (4, 4)

    # Test ShuffleNetV2 forward with widen_factor=1.0, stage_blocks=(4, 8, 4),
    # strides=(2, 2, 1), dilations=(1, 1, 2)
    model = ShuffleNetV2(
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3),
        strides=(2, 2, 1),
        dilations=(1, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 28, 28))
    assert feat[1].shape == torch.Size((1, 232, 14, 14))
    assert feat[2].shape == torch.Size((1, 464, 14, 14))
    assert feat[3].shape == torch.Size((1, 1024, 14, 14))
    assert model.layers[0][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[1][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[2][0].branch2[1].conv.dilation == (2, 2)

    # Test ShuffleNetV2 forward with widen_factor=1.0, stage_blocks=(4, 8, 4),
    # strides=(2, 1, 1), dilations=(1, 2, 4), contract_dilation=True
    model = ShuffleNetV2(
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3),
        strides=(2, 1, 1),
        dilations=(1, 2, 4),
        contract_dilation=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 28, 28))
    assert feat[1].shape == torch.Size((1, 232, 28, 28))
    assert feat[2].shape == torch.Size((1, 464, 28, 28))
    assert feat[3].shape == torch.Size((1, 1024, 28, 28))
    assert model.layers[0][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[1][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[2][0].branch2[1].conv.dilation == (2, 2)

    # Test ShuffleNetV2 forward with widen_factor=1.0, stage_blocks=(4, 8, 4),
    # strides=(1, 1, 1), dilations=(2, 2, 4), contract_dilation=True
    model = ShuffleNetV2(
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3),
        strides=(1, 1, 1),
        dilations=(2, 2, 4),
        contract_dilation=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert is_tuple_of(feat, torch.Tensor)
    assert feat[0].shape == torch.Size((1, 116, 56, 56))
    assert feat[1].shape == torch.Size((1, 232, 56, 56))
    assert feat[2].shape == torch.Size((1, 464, 56, 56))
    assert feat[3].shape == torch.Size((1, 1024, 56, 56))
    assert model.layers[0][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[1][0].branch2[1].conv.dilation == (1, 1)
    assert model.layers[2][0].branch2[1].conv.dilation == (2, 2)
