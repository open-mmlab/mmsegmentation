# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.cityscapes import *
    from .._base_.default_runtime import *
    from .._base_.models.lraspp_m_v3_d8 import *
    from .._base_.schedules.schedule_320k import *

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
# Re-config the data sampler.
model.update(data_preprocessor=data_preprocessor)
train_dataloader.update(batch_size=4, num_workers=4)
val_dataloader.update(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
