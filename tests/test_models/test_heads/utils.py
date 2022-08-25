# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm


def _conv_has_norm(module, sync_bn):
    for m in module.modules():
        if isinstance(m, ConvModule):
            if not m.with_norm:
                return False
            if sync_bn:
                if not isinstance(m.bn, SyncBatchNorm):
                    return False
    return True


def to_cuda(module, data):
    module = module.cuda()
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].cuda()
    return module, data
