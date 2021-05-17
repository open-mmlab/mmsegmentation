import pytest
import torch

from mmseg.models.decode_heads import SETRUPHead
from .utils import to_cuda


def test_setr_up_head(capsys):

    with pytest.raises(AssertionError):
        # num_convs must be [2/4]
        SETRUPHead(num_classes=19, num_convs=-1)

    with pytest.raises(AssertionError):
        # kernel_size must be [1/3]
        SETRUPHead(num_classes=19, kernel_size=2)

    with pytest.raises(TypeError):
        # unutilizing ConvModule, so there is not auto norm layer.
        SETRUPHead(in_channels=32, channels=16, embed_dim=32, num_classes=19)

    with pytest.raises(AssertionError):
        # in_channels must be int type and in_channels must be same
        # as embed_dim.
        SETRUPHead(
            in_channels=(32, 32), channels=16, embed_dim=32, num_classes=19)
        SETRUPHead(in_channels=24, channels=16, embed_dim=32, num_classes=19)

    with pytest.raises(NotImplementedError):
        x = [torch.randn(1, 32, 4, 4)]
        # PUP head
        # when num_convs=4 , num_up_layer must be 4.
        head = SETRUPHead(
            in_channels=32,
            channels=16,
            embed_dim=32,
            num_classes=19,
            num_convs=4,
            num_up_layer=1,
            norm_cfg=dict(type='SyncBN'))
        head(x)

    with pytest.raises(NotImplementedError):
        x = [torch.randn(1, 32, 4, 4)]
        # Naive head
        # when num_convs=2, num_up_layer can be [1/2].
        head = SETRUPHead(
            in_channels=32,
            channels=16,
            num_classes=19,
            embed_dim=32,
            num_convs=2,
            num_up_layer=4,
            norm_cfg=dict(type='SyncBN'))
        head(x)

    # test init_weights of head
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        embed_dim=32,
        norm_cfg=dict(type='SyncBN'),
        num_classes=19)
    head.init_weights()

    # test inference of Naive head
    # the auxiliary head of Naive head is same as Naive head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=2,
        num_up_layer=1,
        kernel_size=1,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size

    # Input square NCHW format feature information
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h, w)

    # Input non-square NCHW format feature information
    x = [torch.randn(1, 32, h, w * 2)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h, w * 2)

    # test inference of PUP head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=4,
        num_up_layer=4,
        kernel_size=3,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    # Input square NCHW format feature information
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 8, w * 8)

    # Input non-square NCHW format feature information
    x = [torch.randn(1, 32, h, w * 2)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, h * 8, w * 16)

    # test inference of PUP auxiliary head
    img_size = 32
    patch_size = 16
    head = SETRUPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=2,
        num_up_layer=2,
        kernel_size=3,
        norm_cfg=dict(type='BN'))

    h, w = img_size // patch_size, img_size // patch_size
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
