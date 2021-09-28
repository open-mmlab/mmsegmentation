_base_ = './bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
