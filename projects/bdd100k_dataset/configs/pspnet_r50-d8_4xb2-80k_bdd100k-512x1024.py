_base_ = [
    '../../../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/datasets/bdd100k.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
custom_imports = dict(
    imports=['projects.bdd100k_dataset.mmseg.datasets.bdd100k'])
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
