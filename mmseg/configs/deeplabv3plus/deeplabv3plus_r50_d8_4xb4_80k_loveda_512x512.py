from mmengine.config import read_base

with read_base():
    from .._base_.datasets.loveda import *
    from .._base_.models.deeplabv3plus_r50_d8 import *
    from .._base_.schedules.schedule_80k import *
    from .._base_.default_runtime import *

crop_size = (512, 512)
data_preprocessor.update(size=crop_size)

model.update(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=7),
    auxiliary_head=dict(num_classes=7))
