_base_ = './bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py'
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005),
    loss_scale=512.)
