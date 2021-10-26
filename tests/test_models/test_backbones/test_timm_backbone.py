# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import TIMMBackbone
from .utils import check_norm_state


def test_timm_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = TIMMBackbone()
        model.init_weights(pretrained=0)

    # Test different norm_layer, can be: 'SyncBN', 'BN2d', 'GN', 'LN', 'IN'
    # Test resnet18 from timm, norm_layer='BN2d'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='BN2d')

    # Test resnet18 from timm, norm_layer='SyncBN'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='SyncBN')

    # Test resnet18 from timm, features_only=True, output_stride=32
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 64, 112, 112))
    assert feats[1] == torch.Size((1, 64, 56, 56))
    assert feats[2] == torch.Size((1, 128, 28, 28))
    assert feats[3] == torch.Size((1, 256, 14, 14))
    assert feats[4] == torch.Size((1, 512, 7, 7))

    # Test resnet18 from timm, features_only=True, output_stride=16
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=16)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 64, 112, 112))
    assert feats[1] == torch.Size((1, 64, 56, 56))
    assert feats[2] == torch.Size((1, 128, 28, 28))
    assert feats[3] == torch.Size((1, 256, 14, 14))
    assert feats[4] == torch.Size((1, 512, 14, 14))

    # Test resnet18 from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 64, 112, 112))
    assert feats[1] == torch.Size((1, 64, 56, 56))
    assert feats[2] == torch.Size((1, 128, 28, 28))
    assert feats[3] == torch.Size((1, 256, 28, 28))
    assert feats[4] == torch.Size((1, 512, 28, 28))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(model_name='efficientnet_b1', pretrained=True)

    # Test resnetv2_50x1_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 32, 32)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 64, 16, 16))
    assert feats[1] == torch.Size((1, 256, 8, 8))
    assert feats[2] == torch.Size((1, 512, 4, 4))
    assert feats[3] == torch.Size((1, 1024, 4, 4))
    assert feats[4] == torch.Size((1, 2048, 4, 4))

    # Test resnetv2_50x3_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x3_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 16, 16)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 192, 8, 8))
    assert feats[1] == torch.Size((1, 768, 4, 4))
    assert feats[2] == torch.Size((1, 1536, 2, 2))
    assert feats[3] == torch.Size((1, 3072, 2, 2))
    assert feats[4] == torch.Size((1, 6144, 2, 2))

    # Test resnetv2_101x1_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_101x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 16, 16)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 64, 8, 8))
    assert feats[1] == torch.Size((1, 256, 4, 4))
    assert feats[2] == torch.Size((1, 512, 2, 2))
    assert feats[3] == torch.Size((1, 1024, 2, 2))
    assert feats[4] == torch.Size((1, 2048, 2, 2))

    # Test resnetv2_101x3_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_101x3_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 16, 16)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 192, 8, 8))
    assert feats[1] == torch.Size((1, 768, 4, 4))
    assert feats[2] == torch.Size((1, 1536, 2, 2))
    assert feats[3] == torch.Size((1, 3072, 2, 2))
    assert feats[4] == torch.Size((1, 6144, 2, 2))

    # Test resnetv2_152x2_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_152x2_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 16, 16)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 128, 8, 8))
    assert feats[1] == torch.Size((1, 512, 4, 4))
    assert feats[2] == torch.Size((1, 1024, 2, 2))
    assert feats[3] == torch.Size((1, 2048, 2, 2))
    assert feats[4] == torch.Size((1, 4096, 2, 2))

    # Test resnetv2_152x4_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_152x4_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 16, 16)
    feats = model(imgs)
    feats = [feat.shape for feat in feats]
    assert len(feats) == 5
    assert feats[0] == torch.Size((1, 256, 8, 8))
    assert feats[1] == torch.Size((1, 1024, 4, 4))
    assert feats[2] == torch.Size((1, 2048, 2, 2))
    assert feats[3] == torch.Size((1, 4096, 2, 2))
    assert feats[4] == torch.Size((1, 8192, 2, 2))
