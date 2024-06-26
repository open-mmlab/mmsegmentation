# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .._base_.datasets.cityscapes import *
    from .._base_.default_runtime import *
    from .._base_.models.lraspp_m_v3_d8 import *
    from .._base_.schedules.schedule_320k import *

checkpoint = 'open-mmlab://contrib/mobilenet_v3_large'
crop_size = (512, 1024)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(
        data_preprocessor=data_preprocessor,
        backbone=dict(
            init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint))))
# Re-config the data sampler.
train_dataloader.update(dict(batch_size=4, num_workers=4))
val_dataloader.update(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader
