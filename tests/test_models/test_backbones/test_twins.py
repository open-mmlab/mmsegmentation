import pytest
import torch

from mmseg.models.backbones.twins import (ALTGVT, PCPVT,
                                          PyramidVisionTransformer)


def test_PyramidVisionTransformer():
    # test alt_gvt structure and forward
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = PyramidVisionTransformer()
        model.init_weights(pretrained=0)


def test_pcpvt():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = PCPVT()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)

    # Test extra_norm = True
    model = PCPVT(extra_norm=True)
    model.train()


def test_altgvt():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = ALTGVT()
    model.init_weights()
    model(imgs)

    model = ALTGVT()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 128, 128)
    model(imgs)

    # Test extra_norm = True
    model = ALTGVT(extra_norm=True)
    model.train()
