import torch.nn as nn
from mmcv.utils.parrots_wrapper import SyncBatchNorm

from mmseg.models.utils import revert_sync_batchnorm


def test_syncbn2bn():
    model = nn.Module()
    model.add_module('SyncBN', SyncBatchNorm(1))
    model = revert_sync_batchnorm(model)

    for m in model.modules():
        if isinstance(m, SyncBatchNorm):
            raise TypeError
