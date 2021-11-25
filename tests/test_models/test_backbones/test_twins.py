import torch

from mmseg.models.backbones.twins import PCPVT, SVT


def test_pcpvt():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 56, 56)
    model = PCPVT()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 32, 32)
    model(imgs)

    # Test norm_after_stage = True
    model = PCPVT(norm_after_stage=True)
    model.train()


def test_svt():
    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 56, 56)
    model = SVT()
    model.init_weights()
    model(imgs)

    model = SVT()
    model.init_weights()
    model(imgs)

    # Test convertible img_size
    imgs = torch.randn(1, 3, 32, 32)
    model(imgs)

    # Test norm_after_stage = True
    model = SVT(norm_after_stage=True)
    model.train()
