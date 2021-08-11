import pytest
import torch

from mmseg.models.decode_heads import SFHead
from .utils import _conv_has_norm, to_cuda
from mmseg.models import ResNet

def test_sf_head():
    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        SFHead(in_channels=512,channels=128,num_classes=19,pool_scales=1)

    # test no norm_cfg
    head = SFHead(in_channels=512, channels=128, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    self = ResNet(depth=18)
    self.eval()
    inputs = torch.rand(1, 3, 1024, 1024)
    resnet_outputs = self.forward(inputs)
    # inputs = [torch.randn(1, 64, 8, 8),
    #           torch.randn(1, 128, 4, 4),
    #           torch.randn(1, 256, 2, 2),
    #           torch.randn(1, 512, 1, 1)]
    head = SFHead(
        in_channels=512, channels=128, num_classes=19, pool_scales=(1, 2, 3, 6))
    # if torch.cuda.is_available():
    #     head, resnet_outputs = to_cuda(head, resnet_outputs)
    outputs = head(resnet_outputs)
    #assert outputs.shape == (1, head.num_classes, 45, 45)