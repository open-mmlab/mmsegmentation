import pytest
import torch

from mmseg.models.decode_heads import PSPHead
from .utils import _conv_has_norm, to_cuda


def test_psp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        PSPHead(in_channels=32, channels=16, num_classes=19, pool_scales=1)

    # test no norm_cfg
    head = PSPHead(in_channels=32, channels=16, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45)]
    head = PSPHead(
        in_channels=32, channels=16, num_classes=19, pool_scales=(1, 2, 3))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.psp_modules[0][0].output_size == 1
    assert head.psp_modules[1][0].output_size == 2
    assert head.psp_modules[2][0].output_size == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
