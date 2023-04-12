# Copyright (c) OpenMMLab. All rights reserved.
import torch

# from mmseg.registry import MODELS


class SAMInferencer:

    def __init__(self, sam_type: str = 'vit-b') -> None:
        assert sam_type in ['vit-b', 'vit-l', 'vit-h']

    def forward(self, image) -> torch.Tensor:
        return image
