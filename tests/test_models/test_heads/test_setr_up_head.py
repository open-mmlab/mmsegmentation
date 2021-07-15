import pytest
import torch

from mmseg.models.decode_heads import SETRUPHead
from .utils import to_cuda


def test_setr_up_head(capsys):

    with pytest.raises(AssertionError):
        # kernel_size must be [1/3]
        SETRUPHead(num_classes=19, kernel_size=2)

    with pytest.raises(AssertionError):
        # in_channels must be int type and in_channels must be same
        # as embed_dim.
        SETRUPHead(in_channels=(32, 32), channels=16, num_classes=19)

    # test init_cfg of head
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        norm_cfg=dict(type='SyncBN'),
        num_classes=19,
        init_cfg=dict(type='Kaiming'))
    super(SETRUPHead, head).init_weights()

    # test inference of Naive head
    # the auxiliary head of Naive head is same as Naive head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        num_convs=1,
        up_scale=4,
        kernel_size=1,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size

    # Input square NCHW format feature information
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 4, w * 4)

    # Input non-square NCHW format feature information
    x = [torch.randn(1, 32, h, w * 2)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 4, w * 8)
