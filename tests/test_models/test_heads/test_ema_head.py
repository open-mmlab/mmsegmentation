import torch

from mmseg.models.decode_heads import EMAHead
from .utils import to_cuda


def test_emanet_head():
    head = EMAHead(
        in_channels=32,
        ema_channels=24,
        channels=16,
        num_stages=3,
        num_bases=16,
        num_classes=19)
    for param in head.ema_mid_conv.parameters():
        assert not param.requires_grad
    assert hasattr(head, 'ema_module')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
