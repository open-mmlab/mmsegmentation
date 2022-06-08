_base_ = './deeplabv3_r101-d8_512x1024_80k_cityscapes.py'
# fp16 settings
default_hooks = dict(optimizer=dict(type='Fp16OptimizerHook', loss_scale=512.))
# fp16 placeholder
fp16 = dict()
