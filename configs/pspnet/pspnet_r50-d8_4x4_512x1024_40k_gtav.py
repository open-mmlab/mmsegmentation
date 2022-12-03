_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/gtav.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
load_from = 'mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'