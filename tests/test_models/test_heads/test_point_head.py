import torch
from mmcv.utils import ConfigDict

from mmseg.models.decode_heads import FCNHead, PointHead
from .utils import to_cuda


def test_point_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    point_head = PointHead(
        in_channels=[32], in_index=[0], channels=16, num_classes=19)
    assert len(point_head.fcs) == 3
    fcn_head = FCNHead(in_channels=32, channels=16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(point_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    test_cfg = ConfigDict(
        subdivision_steps=2, subdivision_num_points=8196, scale_factor=2)
    output = point_head.forward_test(inputs, prev_output, None, test_cfg)
    assert output.shape == (1, point_head.num_classes, 180, 180)
