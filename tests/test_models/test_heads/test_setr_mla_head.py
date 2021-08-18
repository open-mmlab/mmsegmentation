# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import SETRMLAHead
from .utils import to_cuda


def test_setr_mla_head(capsys):

    with pytest.raises(AssertionError):
        # MLA requires input multiple stage feature information.
        SETRMLAHead(in_channels=32, channels=16, num_classes=19, in_index=1)

    with pytest.raises(AssertionError):
        # multiple in_indexs requires multiple in_channels.
        SETRMLAHead(
            in_channels=32, channels=16, num_classes=19, in_index=(0, 1, 2, 3))

    with pytest.raises(AssertionError):
        # channels should be len(in_channels) * mla_channels
        SETRMLAHead(
            in_channels=(32, 32, 32, 32),
            channels=32,
            mla_channels=16,
            in_index=(0, 1, 2, 3),
            num_classes=19)

    # test inference of MLA head
    img_size = (32, 32)
    patch_size = 16
    head = SETRMLAHead(
        in_channels=(32, 32, 32, 32),
        channels=64,
        mla_channels=16,
        in_index=(0, 1, 2, 3),
        num_classes=19,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    # Input square NCHW format feature information
    x = [
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w)
    ]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 4, w * 4)

    # Input non-square NCHW format feature information
    x = [
        torch.randn(1, 32, h, w * 2),
        torch.randn(1, 32, h, w * 2),
        torch.randn(1, 32, h, w * 2),
        torch.randn(1, 32, h, w * 2)
    ]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 4, w * 8)
