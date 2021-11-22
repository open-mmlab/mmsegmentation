# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import FCNHead, OCRHead
from .utils import to_cuda


def test_ocr_head():

    inputs = [torch.randn(1, 8, 23, 23)]
    ocr_head = OCRHead(
        in_channels=8, channels=4, num_classes=19, ocr_channels=8)
    fcn_head = FCNHead(in_channels=8, channels=4, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(ocr_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    output = ocr_head(inputs, prev_output)
    assert output.shape == (1, ocr_head.num_classes, 23, 23)
