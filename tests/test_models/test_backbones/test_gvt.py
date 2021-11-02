import pytest
import torch

from mmseg.models.backbones.gvt import (pcpvt_small_v0, pcpvt_base_v0,
                                        pcpvt_large, alt_gvt_small,
                                        alt_gvt_base, alt_gvt_large,
                                        PyramidVisionTransformer)


def test_PyramidVisionTransformer():
    # test alt_gvt structure and forward
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = PyramidVisionTransformer()
        model.init_weights(pretrained=0)


def test_CVPTV2():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = pcpvt_small_v0()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)

    # Test extra_norm = True
    model = pcpvt_small_v0(extra_norm=True)
    model.train()


def test_ALTGVT():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = alt_gvt_small()
    model.init_weights()
    model(imgs)

    model = pcpvt_base_v0()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)

    # Test extra_norm = True
    model = alt_gvt_small(extra_norm=True)
    model.train()


def test_pcpvt_small():
    # test pcpvt_small_v0 structure and forward



def test_pcpvt_base():
    # test pcpvt_base_v0 structure and forward
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = pcpvt_base_v0()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)



def test_pcpvt_large():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = pcpvt_large()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)



def test_alt_gvt_small():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = alt_gvt_small()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)



def test_alt_gvt_base():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = alt_gvt_base()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)


def test_alt_gvt_large():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = alt_gvt_large()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)


