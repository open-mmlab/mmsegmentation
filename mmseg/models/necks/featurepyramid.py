# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): embedding dimension.
        rescales (list[float]): different sampling multiples were
            used to obtain pyramid features. Default: (4, 2, 1, 0.5).
    """

    def __init__(self, embed_dim, rescales):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)
