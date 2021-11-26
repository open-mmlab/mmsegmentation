# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones.vit import VisionTransformer
from .utils import check_norm_state


def test_vit_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = VisionTransformer()
        model.init_weights(pretrained=0)

    with pytest.raises(TypeError):
        # img_size must be int or tuple
        model = VisionTransformer(img_size=512.0)

    with pytest.raises(TypeError):
        # out_indices must be int ,list or tuple
        model = VisionTransformer(out_indices=1.)

    with pytest.raises(TypeError):
        # test upsample_pos_embed function
        x = torch.randn(1, 196)
        VisionTransformer.resize_pos_embed(x, 512, 512, 224, 224, 'bilinear')

    with pytest.raises(AssertionError):
        # The length of img_size tuple must be lower than 3.
        VisionTransformer(img_size=(224, 224, 224))

    with pytest.raises(TypeError):
        # Pretrained must be None or Str.
        VisionTransformer(pretrained=123)

    with pytest.raises(AssertionError):
        # with_cls_token must be True when output_cls_token == True
        VisionTransformer(with_cls_token=False, output_cls_token=True)

    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = VisionTransformer(img_size=(224, ))
    model.init_weights()
    model(imgs)

    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = VisionTransformer(img_size=(224, 224))
    model(imgs)

    # Test norm_eval = True
    model = VisionTransformer(norm_eval=True)
    model.train()

    # Test ViT backbone with input size of 224 and patch size of 16
    model = VisionTransformer()
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    # Test normal size input image
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test large size input image
    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 16, 16)

    # Test small size input image
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 2, 2)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test unbalanced size input image
    imgs = torch.randn(1, 3, 112, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 7, 14)

    # Test irregular input image
    imgs = torch.randn(1, 3, 234, 345)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 15, 22)

    # Test with_cp=True
    model = VisionTransformer(with_cp=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test with_cls_token=False
    model = VisionTransformer(with_cls_token=False)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test final norm
    model = VisionTransformer(final_norm=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test patch norm
    model = VisionTransformer(patch_norm=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test output_cls_token
    model = VisionTransformer(with_cls_token=True, output_cls_token=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[0][0].shape == (1, 768, 14, 14)
    assert feat[0][1].shape == (1, 768)


def test_vit_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = VisionTransformer(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = VisionTransformer(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = VisionTransformer(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = VisionTransformer(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = VisionTransformer(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = VisionTransformer(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = VisionTransformer(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = VisionTransformer(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = VisionTransformer(pretrained=123, init_cfg=123)
