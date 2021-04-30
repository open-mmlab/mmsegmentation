import torch

from mmseg.models.utils import DropPath


def test_drop_path():

    # zero drop
    layer = DropPath()

    # input NLC format feature
    x = torch.randn((1, 16, 32))
    layer(x)

    # input NLHW format feature
    x = torch.randn((1, 32, 4, 4))
    layer(x)

    # non-zero drop
    layer = DropPath(0.1)

    # input NLC format feature
    x = torch.randn((1, 16, 32))
    layer(x)

    # input NLHW format feature
    x = torch.randn((1, 32, 4, 4))
    layer(x)
