_base_ = './fcn_hr18_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700-1462b75d.pth"
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
