import pytest
import torch

from mmseg.models.decode_heads import SETRUPHead
from .utils import to_cuda


def test_setr_up_head(capsys):

    with pytest.raises(NotImplementedError):
        # num_convs must be [2/4]
        SETRUPHead(num_classes=19, num_convs=-1)

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
            # img_size must be int or tuple.
            SETRUPHead(
                in_channels=32,
                channels=16,
                embed_dim=19,
                num_classes=19,
                norm_cfg=dict(type='SyncBN'),
                img_size=[224, 224])

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

    # test inference of Naive head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        img_size=img_size,
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=2,
        num_up_layer=1,
        conv3x3_conv1x1=False,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size

    # Input NCHW format feature information
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, *img_size)

    # test inference of PUP head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        img_size=img_size,
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=4,
        num_up_layer=4,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    # Input NCHW format feature information
    # NCHW is the main input format.
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, *img_size)

    # test inference of PUP auxiliary head
    img_size = (32, 32)
    patch_size = 16
    head = SETRUPHead(
        img_size=img_size,
        in_channels=32,
        channels=16,
        num_classes=19,
        embed_dim=32,
        num_convs=2,
        num_up_layer=2,
        conv3x3_conv1x1=True,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    # Input NCHW format feature information
    x = [torch.randn(1, 32, h, w)]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, *img_size)
