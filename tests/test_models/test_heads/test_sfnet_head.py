import pytest
import torch

from mmseg.models.decode_heads import SFNetHead
from .utils import _conv_has_norm, to_cuda


def test_sfnet_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        SFNetHead(
            in_channels=2048, channels=256, num_classes=19, pool_scales=1)

    # test no norm_cfg
    head = SFNetHead(in_channels=2048, channels=256, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = SFNetHead(
        in_channels=2048,
        channels=256,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [
        torch.randn(1, 256, 45, 45),
        torch.randn(1, 512, 45, 45),
        torch.randn(1, 1024, 45, 45),
        torch.randn(1, 2048, 45, 45)
    ]
    head = SFNetHead(
        in_channels=2048,
        channels=256,
        num_classes=19,
        pool_scales=(1, 2, 3, 6))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.psp_modules[0][0].output_size == 1
    assert head.psp_modules[1][0].output_size == 2
    assert head.psp_modules[2][0].output_size == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
