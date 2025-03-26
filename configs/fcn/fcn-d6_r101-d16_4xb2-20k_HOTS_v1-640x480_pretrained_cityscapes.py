_base_ = './fcn-d6_r50-d16_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/fcn_d6_r101-d16_512x1024_40k_cityscapes_20210305_130337-9cf2b450.pth"
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
